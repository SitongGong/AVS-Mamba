import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

import fvcore.nn.weight_init as weight_init
from einops import rearrange
from typing import Optional, List

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .cross_modal_mamba import AudioVisionFusion
from .mamba_3d_ import Layer as NormLayer
from .mamba_3d_6 import Layer as AblationLayer


BN_MOMENTUM = 0.1

def get_norm(norm, out_channels): # only support GN or LN
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "GN": lambda channels: nn.GroupNorm(8, channels),
            "LN": lambda channels: nn.LayerNorm(channels)
        }[norm]
    return norm(out_channels)

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(self.norm, torch.nn.SyncBatchNorm), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    
class CrossModalFPN(nn.Module):
    def __init__(self, feature_channels: List, conv_dim: int, mask_dim: int,
                 use_temporal_mamba: bool, use_avfusion: bool, use_cmfpn: bool, 
                 scan_order: int, norm=None):
        super().__init__()
        """
        Args:
            feature_channels: list of fpn feature channel numbers.
            conv_dim: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            dim_feedforward: number of vision-language fusion module ffn channel numbers.
            norm (str or callable): normalization for all conv layers
        """
        self.feature_channels = feature_channels
        self.use_cmfpn = use_cmfpn
        self.use_avfusion = use_avfusion
        self.use_temporal_mamba = use_temporal_mamba
        
        if use_cmfpn and not use_avfusion and not use_temporal_mamba:
            raise ValueError('These options can not be chosen at the same time')

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""     # 通道维度转换 feature_channels: [64, 128, 320, 512]
        for idx, in_channels in enumerate(feature_channels):     # res2 -> res5
            # in_channels: 4x -> 32x
            lateral_norm = get_norm(norm, conv_dim)
            output_norm = get_norm(norm, conv_dim)

            lateral_conv = Conv2d(     # 降维卷积 1 x 1
                    in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
                )
            output_conv = Conv2d(      # 输出卷积 3 x 3
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            weight_init.c2_xavier_fill(lateral_conv)      # 权重初始化
            weight_init.c2_xavier_fill(output_conv)
            stage = idx+1
            self.add_module("adapter_{}".format(stage), lateral_conv)
            self.add_module("layer_{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
            
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]        # res5 -> res2
        self.output_convs = output_convs[::-1]
        # 对最终生成结果进行维度转换
        self.mask_dim = mask_dim
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        weight_init.c2_xavier_fill(self.mask_features)

        # vision-language cross-modal fusion
        # self.audio_pos = PositionEmbeddingSine1D(conv_dim, normalize=True)
        self.self_attns = nn.ModuleList()
        self.cross_attns = nn.ModuleList()
        
        if scan_order != 8:
            for idx in range(len(feature_channels)): # res2 -> res5
                self_attn = AblationLayer(hidden_dim=conv_dim,                  # spatial first
                                drop_path=0.1,
                                use_3d_conv=True,
                                scan_order=scan_order,
                                )
                self.self_attns.append(self_attn)
                cross_attn = AudioVisionFusion(hidden_dim=conv_dim,
                                            audio_to_vision=True,
                                            drop_path=0.1,
                                            )
                self.cross_attns.append(cross_attn)
                
            self.cross_attns = self.cross_attns[::-1]
            self.self_attns = self.self_attns[::-1]

            for m in self.self_attns:
                self._init_weights(m)
            for m in self.cross_attns:
                self._init_weights(m)
        else:
            self.self_attns = nn.ModuleList()
            self.cross_attns = nn.ModuleList()
            
            for idx in range(len(feature_channels)): # res2 -> res5
                # self_attn_ = TemporalBlock(d_model=conv_dim)
                self_attn = NormLayer(hidden_dim=conv_dim,                  # spatial first
                                drop_path=0.1,
                                use_3d_conv=True,
                                )
                cross_attn = AudioVisionFusion(hidden_dim=conv_dim,
                                            audio_to_vision=True,
                                            drop_path=0.1,
                                            )
                self.self_attns.append(self_attn)
                self.cross_attns.append(cross_attn)
                
            self.cross_attns = self.cross_attns[::-1]
            self.self_attns = self.self_attns[::-1]

            for m in self.self_attns:
                self._init_weights(m)
            for m in self.cross_attns:
                self._init_weights(m)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless
        
        Conv2D is not intialized !!!
        """
        # print(m, getattr(getattr(m, "weight", nn.Identity()), "INIT", None), isinstance(m, nn.Linear), "======================")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, audio_feature, memory, num_frame, mask_feature):
        """
        masks: list[bs * T, h, w]
        audio_feature: t, bs, dim
        poses:  list[b*t, dim, h, w]
        memory:  list[batch_size*time, c, hi, wi]
        mask_feature: res2 bs * T, dim, h, w
        """
        # audio_pos = self.audio_pos(audio_feature.permute(1, 2, 0)).permute(2, 0, 1)        # time, bs, dim
        for idx, mem in enumerate(memory[::-1]):    # from res5 -> res3
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cross_attn = self.cross_attns[idx]
            self_attn = self.self_attns[idx]     # 使用双向mamba结构/VMamba结构

            bst, c, h, w = mem.shape
            b = bst // num_frame
            t = num_frame

            vision_features = lateral_conv(mem)  # [b*t, c, h, w]
            if self.use_cmfpn:
                if self.use_temporal_mamba:
                    vision_features = rearrange(vision_features, '(b t) c h w -> b t h w c', t=num_frame)
                    vision_features = self_attn(vision_features)
                    vision_features = rearrange(vision_features, 'b t h w c -> (b t) h w c', t=num_frame)
                else:
                    vision_features = rearrange(vision_features, '(b t) c h w -> b t h w c', t=num_frame)
                    vision_features = rearrange(vision_features, 'b t h w c -> (b t) h w c', t=num_frame)
                
                if self.use_avfusion:
                    cur_fpn = cross_attn(vision_features, audio_feature, num_frame)
                else:
                    cur_fpn = vision_features
            else:
                vision_features = rearrange(vision_features, '(b t) c h w -> b t h w c', t=num_frame)
                vision_features = rearrange(vision_features, 'b t h w c -> (b t) h w c', t=num_frame)
                cur_fpn = vision_features
            cur_fpn = rearrange(cur_fpn, 'bt h w c -> bt c h w')

            # upsample
            if idx == 0: # top layer
                y = output_conv(cur_fpn)
            else:
                # Following FPN implementation, we use nearest upsampling here
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)     # 对于上一层特征进行上采样和相加

        # 4x level       对于res2
        lateral_conv = self.lateral_convs[-1]
        output_conv = self.output_convs[-1]
        cross_attn = self.cross_attns[-1]
        # self_attn = self.self_attns[-1]
            
        n, c, h, w = mask_feature.shape
        b = n // num_frame
        t = num_frame

        vision_features = lateral_conv(mask_feature)  # [b*t, c, h, w]
        if self.use_cmfpn:
            if self.use_temporal_mamba:
                vision_features = rearrange(vision_features, '(b t) c h w -> b t h w c', t=num_frame)
                vision_features = self_attn(vision_features)
                vision_features = rearrange(vision_features, 'b t h w c -> (b t) h w c', t=num_frame)
            else:
                vision_features = rearrange(vision_features, '(b t) c h w -> b t h w c', t=num_frame)
                vision_features = rearrange(vision_features, 'b t h w c -> (b t) h w c', t=num_frame)
            
            if self.use_avfusion:
                cur_fpn = cross_attn(vision_features, audio_feature, num_frame)
            else:
                cur_fpn = vision_features
        else:
            vision_features = rearrange(vision_features, '(b t) c h w -> b t h w c', t=num_frame)
            vision_features = rearrange(vision_features, 'b t h w c -> (b t) h w c', t=num_frame)
            cur_fpn = vision_features
        cur_fpn = rearrange(cur_fpn, 'bt h w c -> bt c h w')

        # Following FPN implementation, we use nearest upsampling here
        y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
        y = output_conv(y)
        return y

    def forward(self, audio_feature, memory, num_frame, mask_feature):
        y = self.forward_features(audio_feature, memory, num_frame, mask_feature)
        return self.mask_features(y)    # 输出维度映射
    
# 之前的模型结构复杂，且参数量较大，尝试基于Mamba设计更简单的结构，在参数量更少的情况下减少冗余的计算，
# 同时取得更好的结果，去除early integration以及对比损失的尝试

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat

from .backbone.pvt import pvt_v2_b5
from .backbone.resnet import B2_ResNet
from .vggish import VGGish

from .mamba_cmfpn_ import CrossModalFPN
from .cross_scale_temporal_v2 import Layer as MultiscaleTemporalEncoder
from .cross_scale_temporal import Layer as ScanAblationEncoder
from .cross_modal_mamba import AudioVisionFusion           # mamba decoder

from .backbone.max_vit import maxvit_base_tf_512


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()

        self.interp = F.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )
        return x

class CompFormer(nn.Module):
    def __init__(self, d_model, vggish_config, audio_dim, num_frame, num_classes, scale_factor, use_vision_backbone, 
                 if_use_encoder, if_use_decoder, use_inter_decoder, use_intra_decoder, use_temporal_encoder, use_spatial_encoder,
                 if_use_cmfpn, use_temporal_mamba, use_avfusion, scan_order, num_feature_level=3, transformer_feat=[1, 2, 3], img_size=224):
        super().__init__()
        self.audio_backbone = VGGish(**vggish_config)
        
        # 推理阶段如下代码已经不重要了，因为在推理阶段所有的模型权重都会重新加载一遍
        if use_vision_backbone == "PVTv2":
            input_shape = {'channel': [64, 128, 320, 512], 'stride': [4, 8, 16, 32]}
            self.vision_backbone = pvt_v2_b5(init_weights_path='/bin/pretrained_backbones/pvt_v2_b5.pth')
        elif use_vision_backbone == "ResNet50":
            input_shape = {'channel': [256, 512, 1024, 2048], 'stride': [4, 8, 16, 32]}
            self.vision_backbone = B2_ResNet(init_weights_path='/bin/pretrained_backbones/resnet50-19c8e357.pth')
        elif use_vision_backbone == "VMamba":
            from .VMamba.vmamba import Backbone_VSSM
            input_shape = {'channel': [128, 256, 512, 1024], 'stride': [4, 8, 16, 32]}
            self.vision_backbone = Backbone_VSSM(pretrained='model/VMamba/vssm_base_0229_ckpt_epoch_237.pth', 
                                                depths=[2, 2, 15, 2], dims=128, drop_path_rate=0.6, 
                                                patch_size=4, in_chans=3, num_classes=1000, 
                                                ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
                                                ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
                                                ssm_init="v0", forward_type="v05_noz", 
                                                mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
                                                patch_norm=True, norm_layer=("ln2d"), 
                                                downsample_version="v3", patchembed_version="v2", 
                                                use_checkpoint=False, posembed=False, imgsize=512)
        elif use_vision_backbone == "Vim":
            from .vim.backbone import VisionMamba
            input_shape = {'channel': [768, 768, 768, 768], 'stride': [4, 8, 16, 32]}
            self.vision_backbone = VisionMamba(img_size=img_size, patch_size=16, embed_dim=768, depth=24, if_fpn=False, pretrained='model/vim/vim_b_midclstok_81p9acc.pth')
            # self.vision_backbone = vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2(pretrained=True)
        elif use_vision_backbone == "MaxVit":
            input_shape = {'channel': [96, 192, 384, 768], 'stride': [4, 8, 16, 32]}
            self.vision_backbone = maxvit_base_tf_512(pretrained=True)
        else:
            raise ValueError("The vision backbone can only be PVTv2, ResNet50 or VMamba")
        
        self.use_vision_backbone = use_vision_backbone
        self.vision_backbone_frozen(unfrozen=False)
        for param in self.audio_backbone.parameters():
            param.requires_grad = False
            
        # These are options for ablation study
        self.use_temporal_encoder = use_temporal_encoder
        self.use_spatial_encoder = use_spatial_encoder
        self.if_use_encoder = if_use_encoder
        
        self.if_use_decoder = if_use_decoder
        self.use_inter_decoder = use_inter_decoder
        self.use_intra_decoder = use_intra_decoder
        
        self.if_use_cmfpn = if_use_cmfpn
        self.use_temporal_mamba = use_temporal_mamba
        self.use_avfusion = use_avfusion
        
        if (scan_order != 8 and not if_use_decoder) or (scan_order != 8 and not if_use_decoder) or (scan_order != 8 and not if_use_cmfpn):
            raise ValueError("The ablation of scan directions cannot be implemented with others")
            
        if not use_temporal_encoder and not use_spatial_encoder and if_use_encoder:
            raise ValueError("These options can not be chosen at the same time")
            
        if not use_intra_decoder and not use_inter_decoder and if_use_decoder:
            raise ValueError("These options can not be chosen at the same time")
            
        # The architecture of the model
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.num_frame = num_frame
        # self.query_embed = nn.Embedding(num_query, d_model)
        self.transformer_feat = transformer_feat
        self.num_feature_level = num_feature_level

        # mamba encoder
        num_encoder_layers = 2       # encoder layer doesn't have an ablation analysis
        self.mamba_encoder = nn.ModuleList()
        
        if scan_order != 8:
             for _ in range(num_encoder_layers):
                 self.mamba_encoder.append(ScanAblationEncoder(
                                            hidden_dim=d_model, 
                                            drop_path=0.1,
                                            use_conv_3d=True,
                                            use_mix_ffn=True,
                                            use_temporal_scale=True,
                                            scan_order=scan_order,
                                            ))
        
        if self.if_use_encoder and scan_order == 8:
            for _ in range(num_encoder_layers):
                if not use_spatial_encoder:
                    self.mamba_encoder.append(MultiscaleTemporalEncoder(
                                            hidden_dim=d_model, 
                                            drop_path=0.1,
                                            use_conv_3d=True,
                                            use_mix_ffn=True,
                                            use_temporal_scale=True,
                                            use_spatial_scale=False, 
                                                )
                                              )
                if not use_temporal_encoder:
                    self.mamba_encoder.append(MultiscaleTemporalEncoder(
                                                hidden_dim=d_model, 
                                                drop_path=0.1,
                                                use_conv_3d=True,
                                                use_mix_ffn=True,
                                                use_temporal_scale=False,
                                                use_spatial_scale=True, 
                                                    )
                                              )
                if use_temporal_encoder and use_spatial_encoder:
                    self.mamba_encoder.append(MultiscaleTemporalEncoder(
                                                hidden_dim=d_model, 
                                                drop_path=0.1,
                                                use_conv_3d=True,
                                                use_mix_ffn=True,
                                                use_temporal_scale=True,
                                                use_spatial_scale=True, 
                                                    )
                                              )
        
        # mamba decoder
        num_layers = 2
        self.num_layers = num_layers
        self.mamba_decoder = nn.ModuleList()
        for _ in range(num_layers):
            self.mamba_decoder_block = nn.ModuleList()
            if self.if_use_decoder:
                for _ in range(num_feature_level):
                    mamba_decoder_block = nn.ModuleList()
                    intra_frame_interaction = AudioVisionFusion(
                                                hidden_dim=d_model,
                                                drop_path=0.1,
                                                audio_to_vision=False,
                                                inter_frame=False,
                                                )
                    mamba_decoder_block.append(intra_frame_interaction)
                    inter_frame_interaction = AudioVisionFusion(
                                                hidden_dim=d_model,
                                                drop_path=0.1,
                                                audio_to_vision=False,
                                                inter_frame=True,
                                                )
                    mamba_decoder_block.append(inter_frame_interaction)
                
                    self.mamba_decoder_block.append(mamba_decoder_block)
            self.mamba_decoder.append(self.mamba_decoder_block)
        
        input_proj_list = []   # 对backbone输出的每一层建立维度投影
        for i in input_shape['channel']:        # res2 -> res5四层特征图
            input_proj_list.append(
                nn.Sequential(nn.Conv2d(i, d_model, kernel_size=1),
                              nn.GroupNorm(32, d_model))
            )
            
        self.input_proj = nn.ModuleList(input_proj_list) 

        feature_channels = 4 * [d_model]
        self.cmfpn = CrossModalFPN(feature_channels=feature_channels,
                                   conv_dim=d_model,
                                   mask_dim=d_model,
                                   use_temporal_mamba=use_temporal_mamba,
                                   use_avfusion=use_avfusion,
                                   use_cmfpn=if_use_cmfpn,
                                   scan_order=scan_order)

        # post processing
        self.mlp = FFN(1, 2048, d_model, 3)
        self.fc = nn.Sequential(
            nn.Conv2d(d_model, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=scale_factor, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        )
        
    def vision_backbone_frozen(self, unfrozen=False):
        for param in self.vision_backbone.parameters():
            param.requires_grad = unfrozen

    def forward(self, audio_feat, vision_feat):
        with torch.no_grad():
            audio_feat = self.audio_backbone(audio_feat)
        audio_feat = self.audio_proj(audio_feat)        # bs * T, dim

        vision_feat = self.vision_backbone(vision_feat)
        # res2层特征图，不参与deformable attn 计算，和音频进行两次特征交互
        if self.use_vision_backbone == 'MaxVit':
            vision_feat = vision_feat[1: ]
        mask_feature = self.input_proj[0](vision_feat[0])   # bs * T, dim, h, w
        mask = torch.zeros((mask_feature.size(0), mask_feature.size(2), mask_feature.size(3)), device=mask_feature.device, dtype=torch.bool)
        # mask_feature = rearrange(mask_feature, 'bt c h w -> bt h w c')
        audio_feature = rearrange(audio_feat, '(b t) c -> b t c', t=self.num_frame)
        audio_feat = rearrange(audio_feat, '(b t) c -> b t c', t=self.num_frame)
        num_frame = audio_feat.shape[1]
        
        srcs = []
        masks = []
        for i in self.transformer_feat:
            src = self.input_proj[i](vision_feat[i])
            mask = torch.zeros((src.size(0), src.size(2), src.size(3)), device=src.device, dtype=torch.bool)   # b * t, h, w
            srcs.append(src)
            masks.append(mask)
        
        # 利用audio_feature为每帧生成num_query个query
        audio_feat = rearrange(audio_feat, 'b t c -> (b t) c')
        audio_out = audio_feat.unsqueeze(1)     # bs * t, 1, dim

        vis_feat1, vis_feat2, vis_feat3 = rearrange(srcs[0], 'b c h w -> b h w c'), rearrange(srcs[1], 'b c h w -> b h w c'), rearrange(srcs[2], 'b c h w -> b h w c')
        
        if self.if_use_encoder:
            for _, layer in enumerate(self.mamba_encoder):
                vis_feat1, vis_feat2, vis_feat3 = layer(vis_feat1, vis_feat2, vis_feat3, self.num_frame)
        srcs = [vis_feat1, vis_feat2, vis_feat3]      # res3 -> res5: bs * t, h, w, dim

        hs = []
        
        for _, blocks in enumerate(self.mamba_decoder):
            if self.if_use_decoder:
                for idx, layer in enumerate(blocks):
                    if self.use_intra_decoder:
                        audio_out = layer[0](srcs[idx], audio_out, self.num_frame)       # intra-frame interaction
                    if self.use_inter_decoder:
                        audio_out = layer[1](srcs[idx], audio_out, self.num_frame)       # inter-frame interaction
            hs.append(audio_out)
        
        memory = [vis_feat1.permute(0, 3, 1, 2), vis_feat2.permute(0, 3, 1, 2), vis_feat3.permute(0, 3, 1, 2)]
        # srcs = [vis_feat1, vis_feat2, vis_feat3]      # res3 -> res5: bs * t, h, w, dim
        
        mask_features = self.cmfpn(audio_feature=audio_feature, 
                                   memory=memory,
                                   num_frame=self.num_frame,
                                   mask_feature=mask_feature)                                   
        """
        masks: list[bs * T, h, w]
        audio_feature: t, bs, dim
        poses:  list[b*t, dim, h, w]
        memory:  list[batch_size*time, c, hi, wi]
        mask_feature: res2 bs * T, dim, h, w
        """
        # 使用动态卷积层生成中间以及最终mask
        outputs_seg_masks = []
        for lvl in range(len(hs)):
            output_seg_mask = torch.einsum('bchw,blc -> blhw', mask_features, hs[lvl])
            output_seg_mask = self.mlp(output_seg_mask) + mask_features
            outputs_seg_mask = self.fc(output_seg_mask)     # bs * t, nclass=1, h, w
            outputs_seg_masks.append(outputs_seg_mask)

        return outputs_seg_masks, mask_features


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class FFN(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Conv2d(n, k, kernel_size=1, stride=1, padding=0)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

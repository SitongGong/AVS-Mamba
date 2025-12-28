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
# from .cross_scale_mamba import Layer          # mamba encoder
# from .cross_scale_temporal import Layer         # 帧内跨尺度特征交互与帧间跨尺度特征交互相结合
from .cross_scale_temporal_v2 import Layer          # 帧间尺度内特征交互与帧间跨尺度特征交互相结合
# from .cross_scale_spatial import Layer
# from .cross_scale_fusion import Layer
from .cross_modal_mamba import AudioVisionFusion           # mamba decoder
# from .cross_modal_mamba_v1 import CrossModalLayer


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
    def __init__(self, d_model, nheads, num_query, vggish_config, audio_dim, num_frame, mask_dim, dim_feedforward, num_classes, scale_factor, vision_checkpoint, controller_layers, dynamic_mask_channels, num_feature_level=3, rel_coord=True, transformer_feat=[1, 2, 3]):
        super().__init__()
        self.audio_backbone = VGGish(**vggish_config)
        self.vision_backbone = pvt_v2_b5(init_weights_path=vision_checkpoint)
        # self.vision_backbone = B2_ResNet(vision_checkpoint)
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.num_frame = num_frame
        self.query_embed = nn.Embedding(num_query, d_model)
        self.transformer_feat = transformer_feat
        
        # audio reconstruction head
        self.audio_feature_projection = MLP(d_model, d_model, d_model, num_layers=3)
        self.audio_recons_proj = MLP(d_model, d_model, d_model, num_layers=3)

        self.num_feature_level = num_feature_level

        # mamba encoder
        num_encoder_layers = 2
        self.mamba_encoder = nn.ModuleList()
        for _ in range(num_encoder_layers):
            self.mamba_encoder.append(Layer(hidden_dim=d_model, 
                                            # use_cross_scale=True,
                                            drop_path=0.1,
                                            use_conv_3d=True,
                                            use_mix_ffn=True,
                                            use_temporal_scale=True,
                                            ))
        
        # mamba decoder
        num_layers = 2
        self.num_layers = num_layers
        self.mamba_decoder = nn.ModuleList()
        for _ in range(num_layers):
            self.mamba_decoder_block = nn.ModuleList()
            for _ in range(num_feature_level):
                mamba_decoder_block = nn.ModuleList()
                
                intra_frame_interaction = AudioVisionFusion(
                                            hidden_dim=d_model,
                                            drop_path=0.1,
                                            audio_to_vision=False,
                                            inter_frame=False,
                                            )
                inter_frame_interaction = AudioVisionFusion(
                                            hidden_dim=d_model,
                                            drop_path=0.1,
                                            audio_to_vision=False,
                                            inter_frame=True,
                                            )
                mamba_decoder_block.append(intra_frame_interaction)
                mamba_decoder_block.append(inter_frame_interaction)
                
                self.mamba_decoder_block.append(mamba_decoder_block)
            self.mamba_decoder.append(self.mamba_decoder_block)

        self.vision_backbone_frozen(frozen=False)
        for param in self.audio_backbone.parameters():
            param.requires_grad = False

        # 默认由backbone中提取4个尺度特征   res2 -> res5
        # input_shape = {'channel': [256, 512, 1024, 2048], 'stride': [4, 8, 16, 32]}
        input_shape = {'channel': [64, 128, 320, 512], 'stride': [4, 8, 16, 32]}
        
        input_proj_list = []   # 对backbone输出的每一层建立维度投影
        for i in input_shape['channel']:        # res2 -> res5四层特征图
            input_proj_list.append(
                nn.Sequential(nn.Conv2d(i, d_model, kernel_size=1),
                              nn.GroupNorm(32, d_model))
            )
            
        # 对最后一层feature map进行下采样
        if num_feature_level > len(transformer_feat):
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(i, d_model, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, d_model)
                )
            )
            
        self.input_proj = nn.ModuleList(input_proj_list) 

        feature_channels = 4 * [d_model]
        self.cmfpn = CrossModalFPN(feature_channels=feature_channels,
                                    conv_dim=d_model,
                                    mask_dim=d_model)

        self.mlp = FFN(1, 2048, d_model, 3)
        self.fc = nn.Sequential(
            nn.Conv2d(d_model, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=scale_factor, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        )
        
    def vision_backbone_frozen(self, frozen=False):
        for param in self.vision_backbone.parameters():
            param.requires_grad = frozen

    def mul_temporal_mask(self, feats, vid_temporal_mask_flag=None):
        if vid_temporal_mask_flag is None:
            return feats
        else:
            if isinstance(feats, list):
                out = []
                for x in feats:
                    out.append(x * vid_temporal_mask_flag)
            elif isinstance(feats, torch.Tensor):
                out = feats * vid_temporal_mask_flag

            return out

    def forward(self, audio_feat, vision_feat, target_mask, vid_temporal_mask_flag=None):
        if vid_temporal_mask_flag is not None:
            vid_temporal_mask_flag = vid_temporal_mask_flag.view(-1, 1, 1, 1)

        with torch.no_grad():
            audio_feat = self.audio_backbone(audio_feat)
        audio_feat = self.audio_proj(audio_feat)        # bs * T, dim
        origin_audio_feat = audio_feat

        vision_feat = self.vision_backbone(vision_feat)
        vision_feat = self.mul_temporal_mask(vision_feat, vid_temporal_mask_flag)        # 若为MS3或S4数据集中图像，则只抽取前5帧

        # res2层特征图，不参与deformable attn 计算，和音频进行两次特征交互
        mask_feature = self.input_proj[0](vision_feat[0])   # bs * T, dim, h, w
        mask = torch.zeros((mask_feature.size(0), mask_feature.size(2), mask_feature.size(3)), device=mask_feature.device, dtype=torch.bool)
        # pos_embed = self.positional_encoding(mask)    # batch_size*time, c, hi, wi
        # h, w = pos_embed.shape[-2: ]
        # pos_embed = rearrange(pos_embed, '(b t) c h w -> (t h w) b c', t=self.num_frame)
        mask_feature = rearrange(mask_feature, 'bt c h w -> bt h w c')
        audio_feature = rearrange(audio_feat, '(b t) c -> b t c', t=self.num_frame)
        audio_feat = rearrange(audio_feat, '(b t) c -> b t c', t=self.num_frame)

        # audio_feat = self.vision_to_audio_fusion(mask_feature, audio_feat, self.num_frame)
        # mask_feature = self.audio_to_vision_fusion(mask_feature, audio_feature, self.num_frame)
        mask_feature = rearrange(mask_feature, 'bt h w c -> bt c h w')      # (b t) c h w
        # res2_pos_embed = rearrange(pos_embed, '(t h w) b c -> (b t) c h w', t=self.num_frame, h=h, w=w)
        
        srcs = []
        masks = []
        pos_embeds = []
        for i in self.transformer_feat:
            src = self.input_proj[i](vision_feat[i])
            mask = torch.zeros((src.size(0), src.size(2), src.size(3)), device=src.device, dtype=torch.bool)   # b * t, h, w
            # pos_embed = self.positional_encoding(mask)  # (b t) c h w
            # , w = pos_embed.shape[-2: ]
    
            # pos_embed = rearrange(pos_embed, '(b t) c h w -> (t h w) b c', t=self.num_frame)
            src = rearrange(src, 'bt c h w -> bt h w c')
            # audio_feat = self.vision_to_audio_fusion(src, audio_feat, self.num_frame)
            # src = self.audio_to_vision_fusion(src, audio_feature, self.num_frame)
            src = rearrange(src, 'bt h w c -> bt c h w')

            srcs.append(src)
            # pos_embed = rearrange(pos_embed, '(t h w) b c -> (b t) c h w', t=self.num_frame, h=h, w=w)
            # pos_embeds.append(pos_embed)
            masks.append(mask)
        ##############对比损失修改的地方#################  
        # audio_feat_rec = self.audio_feature_projection(audio_feat)          # time, bs, dim
        # audio_feat_rec = audio_feat_rec.flatten(0, 1)                # time * bs, dim
        
        """
        Deformable Transformer
        srcs (list[Tensor]): list of tensors num_layers x [batch_size*time, c, hi, wi], input of encoder
        tgt (Tensor): [batch_size, time, num_queries_per_frame, c]
        masks (list[Tensor]): list of tensors num_layers x [batch_size*time, hi, wi], the mask of srcs
        pos_embeds (list[Tensor]): list of tensors num_layers x [batch_size*time, c, hi, wi], position encoding of srcs
        query_embed (Tensor): [num_queries, c]
        """
        # 利用audio_feature为每帧生成num_query个query
        audio_feat = rearrange(audio_feat, 'b t c -> (b t) c')
        audio_out = audio_feat.unsqueeze(1)     # bs * t, 1, dim

        vis_feat1, vis_feat2, vis_feat3 = rearrange(srcs[0], 'b c h w -> b h w c'), rearrange(srcs[1], 'b c h w -> b h w c'), rearrange(srcs[2], 'b c h w -> b h w c')
        for _, layer in enumerate(self.mamba_encoder):
            vis_feat1, vis_feat2, vis_feat3 = layer(vis_feat1, vis_feat2, vis_feat3, self.num_frame)
        srcs = [vis_feat1, vis_feat2, vis_feat3]      # res3 -> res5: bs * t, h, w, dim

        hs = []
        for _, blocks in enumerate(self.mamba_decoder):
            for idx, layer in enumerate(blocks):
                audio_out = layer[0](srcs[idx], audio_out, self.num_frame)       # intra-frame interaction
                audio_out = layer[1](srcs[idx], audio_out, self.num_frame)       # inter-frame interaction
            hs.append(audio_out)

        memory = [vis_feat1.permute(0, 3, 1, 2), vis_feat2.permute(0, 3, 1, 2), vis_feat3.permute(0, 3, 1, 2)]
        
        # tgt = self.query_generator(audio_feat.unsqueeze(1))    # bs * t, num_query, dim
        # tgt = rearrange(tgt, '(b t) n c -> b t n c', t=self.num_frame)    # batch_size, time, num_queries_per_frame, dim
        # query_embed = self.query_embed.weight      # num_query, dim  可学习位置嵌入
        # 进行多尺度图像特征的融合（res3 -> res5）以及图像和音频对应帧之间的交互
        # hs, memory, _, inter_references, _, _, _, mti_output = self.deformable_transformer(srcs, tgt, masks, pos_embeds, query_embed)
        # hs: [l, batch_size*time, num_queries_per_frame, c], where l is number of decoder layers
        # init_reference_out: [batch_size*time, num_queries_per_frame, 2]
        # inter_references_out: [l, batch_size*time, num_queries_per_frame, 4]
        # memory: [batch_size*time, \sigma(hi*wi), c]
        # memory_features: list[Tensor]

        # masks = masks[: -1]                # res3 -> res5
        # pos_embeds = pos_embeds[: -1]         # res3 -> res5
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

        outputs_seg_masks = self.mul_temporal_mask(outputs_seg_masks, vid_temporal_mask_flag)
        mask_features = self.mul_temporal_mask(mask_features, vid_temporal_mask_flag)

        audio_feat_rec = self.audio_feature_projection(audio_out.squeeze(1))          # bs * time, dim
        # audio_feat_rec = audio_feat_rec.flatten(0, 1)                # time * bs, dim

        origin_audio_feat = self.audio_recons_proj(origin_audio_feat)
        origin_audio_feat = origin_audio_feat.reshape(-1, self.num_frame, origin_audio_feat.shape[-1])
        audio_feat_rec = audio_feat_rec.reshape(-1, self.num_frame, audio_feat_rec.shape[-1])

        return outputs_seg_masks, mask_features, origin_audio_feat, audio_feat_rec


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
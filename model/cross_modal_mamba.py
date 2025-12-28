import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from typing import Optional, Union, Type, List, Tuple, Callable, Dict
from functools import partial

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from causal_conv1d import causal_conv1d_fn

import math


class Vision_to_Audio_Fusion(nn.Module):
    def __init__(self,
                d_model,
                d_state=16,
                d_conv_1d=4,
                d_conv=3,
                expand=2,
                dt_rank="auto",
                dt_min=0.001,
                dt_max=0.1,
                dt_init="random",
                dt_scale=1.0,
                dt_init_floor=1e-4,
                dropout=0.,
                conv_bias=True,
                inter_frame=False,
                bias=False,
                device=None,
                dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        '''
        通过VMamba结构实现跨模态特征交互，首先进行帧内交互，而后进行帧间交互，这一过程中不破坏音频和视频的时序关系
        '''
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.inter_frame = inter_frame

        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # mamba block内部结构 (针对视觉模态)
        self.in_proj_vis = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act_v = nn.SiLU()

        # 针对音频模态
        self.in_proj_audio = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv_1d,
            groups=self.d_inner,
            padding=d_conv_1d - 1,
            **factory_kwargs,
        )
        self.K = 4

        # SSM1
        self.x_proj = [nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs) for i in range(self.K)]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = [self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs) for i in range(self.K)]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.K, merge=True) # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        # Mamba后处理
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
            
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def intra_frame_interaction(self, vis_feat: torch.Tensor, audio_query: torch.Tensor):
        BT, C, H, W = vis_feat.shape
        L = H * W
        K = self.K

        x_hwwh = torch.stack([vis_feat.view(BT, -1, L), torch.transpose(vis_feat, dim0=2, dim1=3).contiguous().view(BT, -1, L)], dim=1).view(BT, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        audio_query = causal_conv1d_fn(audio_query, rearrange(self.conv1d.weight, "d 1 w -> d w"), self.conv1d.bias, activation="silu")       # bs * t, dim, 1
        audio_query = repeat(audio_query, 'bt d l -> bt k d l', k=self.K)
        xs = torch.cat((xs, audio_query), dim=-1)     # bt, k, d, l + 1     将audio query插入到最后
        
        L = L + 1
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(BT, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(BT, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(BT, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(BT, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(BT, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(BT, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(BT, K, -1, L)
        assert out_y.dtype == torch.float
        
        audio_query = out_y[:, :, :, -1]           # bs * t, 4, dim

        return torch.sum(audio_query, dim=1).unsqueeze(-1)     # bs * t, dim, 1

    def inter_frame_interaction(self, vis_feat: torch.Tensor, audio_query: torch.Tensor, num_frame: int):
        B, C, H, W = vis_feat.shape
        L = H * W * num_frame
        K = self.K

        vis_feat = rearrange(vis_feat, '(b t) c h w -> b c t h w', t=num_frame)
        B = B // num_frame

        x_hwwh = torch.stack([vis_feat.reshape(B, C, L), torch.transpose(vis_feat, dim0=3, dim1=4).contiguous().reshape(B, C, L)], dim=1).view(B, 2, C, L)   # bs, 2, dim, thw
        audio_query = causal_conv1d_fn(audio_query, rearrange(self.conv1d.weight, "d 1 w -> d w"), self.conv1d.bias, activation="silu")     # bs, dim, t
        audio_query = repeat(audio_query, 'b d t -> b k d t', k=2)     # bs, k, d, t
        x_hwwh1 = torch.cat((x_hwwh, audio_query), dim=-1)        # bs, k, d, thw + t   
        
        x_hwwh_ = torch.flip(x_hwwh, dims=[-1])       # bs, k, dim, thw
        audio_query = torch.flip(audio_query, dims=[-1])        # bs, k, dim, t
        x_hwwh2 = torch.cat((x_hwwh_, audio_query), dim=-1)      # bs, k, dim, thw + t
        xs = torch.cat([x_hwwh1, x_hwwh2], dim=1)      # bs, 4, dim, thw + t

        L = L + num_frame

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        audio_out = out_y[:, :, :, L-num_frame:]     # bs, k, dim, t
        audio_out1 = audio_out[:, 0, ...] + audio_out[:, 1, ...]     # bs, dim, t
        audio_out2 = audio_out[:, 2, ...] + audio_out[:, 3, ...]     # bs, dim, t
        audio_out = audio_out1 + torch.flip(audio_out2, dims=[-1])       # bs, dim, t

        return audio_out     # bs, dim, t

    def forward(self, vis_feat: torch.Tensor, audio_query: torch.Tensor, num_frame: int):
        '''
        vis_feat: bs * t, h, w, dim
        audio_query: bs * t, 1, dim
        '''
        # visual feature
        vis_feat = self.in_proj_vis(vis_feat)
        vis_feat = vis_feat.permute(0, 3, 1, 2).contiguous()      # bs * t, dim, h, w
        # vis_feat = vis_feat.permute(0, 3, 1, 2).contiguous()
        vis_feat = self.act_v(self.conv2d(vis_feat)) # (b, d, h, w)
        
        # audio feature
        audio_query = self.in_proj_audio(audio_query)
        audio, z = audio_query.chunk(2, dim=-1)    # bs, t, dim

        # cross modal interaction
        if self.inter_frame:
            audio = rearrange(audio, '(b t) l c -> b (t l) c', t=num_frame)
            audio_out = self.inter_frame_interaction(vis_feat, audio.transpose(1, 2), num_frame)        # bs, dim, t
            audio_out = rearrange(audio_out, 'b d t -> (b t) d').unsqueeze(-1)        # bs * t, dim, 1
        else:
            audio_out = self.intra_frame_interaction(vis_feat, audio.transpose(1, 2))
        
        # post processing
        audio_out = torch.transpose(audio_out, dim0=1, dim1=2).contiguous()     # bs * t, 1, dim
        audio_out = self.out_norm(audio_out)
        out = audio_out * F.silu(z)

        out = self.out_proj(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class Audio_to_Vision_Fusion(nn.Module):
    def __init__(self,
                d_model,
                d_state=16,
                d_conv_1d=4,
                d_conv=3,
                expand=2,
                dt_rank="auto",
                dt_min=0.001,
                dt_max=0.1,
                dt_init="random",
                dt_scale=1.0,
                dt_init_floor=1e-4,
                dropout=0.,
                conv_bias=True,
                bias=False,
                device=None,
                dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        '''
        通过VMamba结构实现跨模态特征交互，首先进行帧内交互，而后进行帧间交互，这一过程中不破坏音频和视频的时序关系
        '''
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state

        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # mamba block内部结构 (针对视觉模态)
        self.in_proj_vis = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act_v = nn.SiLU()

        # 针对音频模态
        self.in_proj_audio = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv_1d,
            groups=self.d_inner,
            padding=d_conv_1d - 1,
            **factory_kwargs,
        )
        self.K = 4

        # SSM1
        self.x_proj = [nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs) for i in range(self.K)]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = [self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs) for i in range(self.K)]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.K, merge=True) # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        # Mamba后处理
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
            
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def audio_to_vision_fusion(self, vis_feat: torch.Tensor, audio_query: torch.Tensor, num_frame: int):
        B, C, H, W = vis_feat.shape
        L = H * W * num_frame
        K = self.K

        vis_feat = rearrange(vis_feat, '(b t) c h w -> b c t h w', t=num_frame)
        B = B // num_frame

        # 将vis_feat按两种顺序展平
        x_hwwh = torch.stack([vis_feat.reshape(B, C, L), torch.transpose(vis_feat, dim0=3, dim1=4).contiguous().reshape(B, C, L)], dim=1).reshape(B, 2, C, L)   # bs, 2, dim, thw
        audio_query = causal_conv1d_fn(audio_query, rearrange(self.conv1d.weight, "d 1 w -> d w"), self.conv1d.bias, activation="silu")     # bs, dim, t
        audio_query = repeat(audio_query, 'b d t -> b k d t', k=2)     # bs, k, d, t
        x_hwwh1 = torch.cat((audio_query, x_hwwh), dim=-1)        # bs, k, d, t + thw  
        
        x_hwwh_ = torch.flip(x_hwwh, dims=[-1])       # bs, k, dim, thw
        audio_query = torch.flip(audio_query, dims=[-1])        # bs, k, dim, t
        x_hwwh2 = torch.cat((audio_query, x_hwwh_), dim=-1)      # bs, k, dim, t + thw
        xs = torch.cat([x_hwwh1, x_hwwh2], dim=1)      # bs, 4, dim, t + thw

        L = L + num_frame

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        out_y = out_y[:, :, :, num_frame: ]     # bs, k, dim, h * w * t
        L = L - num_frame
        out_y = out_y[:, 0:2] + torch.flip(out_y[:, 2:4], dims=[-1])       # bs, 2, dim, hwt
        out_y = out_y[:, 0].reshape(B, C, num_frame, H, W) + out_y[:, 1].reshape(B, C, num_frame, W, H).transpose(3, 4)    # bs, dim, t, h, w
        out_y = rearrange(out_y, 'b c t h w -> (b t) h w c')
        
        return out_y

    def forward(self, vis_feat: torch.Tensor, audio_query: torch.Tensor, num_frame: int):
        '''
        vis_feat: bs * t, h, w, dim
        audio_query: bs, t, dim
        '''
        # visual feature
        BT, H, W, C = vis_feat.shape
        vis_feat = self.in_proj_vis(vis_feat)
        vis_feat, z = vis_feat.chunk(2, dim=-1)         # bs * t, h, w, dim
        vis_feat = vis_feat.permute(0, 3, 1, 2).contiguous()      # bs * t, dim, h, w
        # vis_feat = vis_feat.permute(0, 3, 1, 2).contiguous()
        vis_feat = self.act_v(self.conv2d(vis_feat)) # (b, d, h, w)

        # audio feature
        audio_query = self.in_proj_audio(audio_query)

        # cross modal interaction
        vis_out = self.audio_to_vision_fusion(vis_feat, audio_query.transpose(1, 2), num_frame)
        
        # post processing
        vis_out = self.out_norm(vis_out)
        out = vis_out * F.silu(z)

        out = self.out_proj(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class MixFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)

        self.norm = nn.LayerNorm(out_features)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = rearrange(x, 'bt h w c -> bt c h w')
        x = self.dwconv(x)
        x = rearrange(x, 'bt c h w -> bt h w c')
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AudioVisionFusion(nn.Module):
    def __init__(self,
                 hidden_dim: int = 0,
                 audio_to_vision: bool=False,
                 drop_path: float = 0,
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 attn_drop_rate: float = 0,
                 d_state: int = 16,
                 mlp_ratio: int = 4, 
                 inter_frame: bool=False,
                 d_conv_1d: int=4
                 ):
        super().__init__()
        self.norm_v = norm_layer(hidden_dim)
        self.norm_a = norm_layer(hidden_dim)
        self.drop_path = DropPath(drop_path)
        self.audio_to_vision = audio_to_vision

        if audio_to_vision:
            self.audio_to_vision_layer = Audio_to_Vision_Fusion(d_model=hidden_dim,
                                                                d_state=d_state,
                                                                dropout=attn_drop_rate,
                                                                d_conv_1d=d_conv_1d,
                                                                )
        else:
            self.vision_to_audio_layer = Vision_to_Audio_Fusion(d_model=hidden_dim,
                                                                d_state=d_state,
                                                                dropout=attn_drop_rate,
                                                                inter_frame=inter_frame,
                                                                d_conv_1d=d_conv_1d,
                                                                )
        
        # self.norm = norm_layer(hidden_dim)
        # self.mix_ffn = MixFFN(in_features=hidden_dim,
        #                       hidden_features=int(mlp_ratio * hidden_dim),
        #                       out_features=hidden_dim)
        
        # self.norm_ = norm_layer(hidden_dim)
        
    def forward(self, vis_feat, audio_query, num_frame):
        '''
        vis_feat: bs * t, h, w, dim
        audio_query: bs, t, dim
        '''
        vis_feat_ = self.norm_v(vis_feat)
        audio_query_ = self.norm_a(audio_query)

        if self.audio_to_vision:
            vis_out = self.audio_to_vision_layer(vis_feat_, audio_query_, num_frame)
            output = self.drop_path(vis_out) + vis_feat
            # output = self.drop_path(self.mix_ffn(self.norm(output))) + output
        else:
            audio_out = self.vision_to_audio_layer(vis_feat_, audio_query_, num_frame)
            output = self.drop_path(audio_out) + audio_query
            # output = self.drop_path(self.gated_mlp(self.norm_(output))) + output

        return output

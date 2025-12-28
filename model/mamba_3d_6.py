import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from typing import Optional, Union, Type, List, Tuple, Callable, Dict
from functools import partial

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

import math


class Mamba3D(nn.Module):
    def __init__(self,
                d_model,
                d_state=16,
                # d_state="auto", # 20240109
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
                use_conv_3d=True,
                scan_order=8,
                device=None,
                dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        '''
        假设不改变时序维度的方向，即在第一阶段只进行空间维度的转换，第二阶段叠加音频模态
        '''
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.use_conv_3d = use_conv_3d

        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # mamba block内部结构
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        if use_conv_3d:
            self.conv3d = nn.Conv3d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=(1, d_conv, d_conv),
            padding=(0, (d_conv - 1) // 2, (d_conv - 1) // 2),
            **factory_kwargs,
        )
        else:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            
        self.act = nn.SiLU()
        self.K = scan_order
        self.scan_order = scan_order

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
    
    def forward_core(self, x: torch.Tensor):
        B, C, T, H, W = x.shape
        L = H * W * T
        K = self.K

        # 这里先使用8个方向的交互，如果会显著增加训练时间的话就换成6个方向       spatial first
        x1 = x.reshape(B, C, -1)                               # bs, dim, thw
        xs1 = torch.stack([x1, torch.flip(x1, dims=[-1])], dim=1)        # bs, 2, dim, thw          

        x2 = x.permute(0, 1, 2, 4, 3).reshape(B, C, -1)        # bs, dim, twh
        xs2 = torch.stack([x2, torch.flip(x2, dims=[-1])], dim=1)        # bs, 2, dim, twh

        x3 = x.permute(0, 1, 3, 4, 2).reshape(B, C, -1)        # bs, dim, hwt
        xs3 = torch.stack([x3, torch.flip(x3, dims=[-1])], dim=1)        # bs, 2, dim, hwt

        x4 = x.permute(0, 1, 4, 3, 2).reshape(B, C, -1)        # bs, dim, wht
        xs4 = torch.stack([x4, torch.flip(x4, dims=[-1])], dim=1)        # bs, 2, dim, wht
        
        x5 = x.permute(0, 1, 3, 2, 4).reshape(B, C, -1)        # bs, dim, htw  
        xs5 = torch.stack([x5, torch.flip(x5, dims=[-1])], dim=1)        # bs, 2, dim, htw  
        
        x6 = x.permute(0, 1, 4, 2, 3).reshape(B, C, -1)        # bs, dim, wth
        xs6 = torch.stack([x6, torch.flip(x6, dims=[-1])], dim=1)        # bs, 2, dim, wth

        # x4 = x.reshape(B, C, -1)          # bs, dim, thw
        # xs4 = torch.stack([x4, torch.flip(x4, dims=[-1])], dim=1)        # bs, 2, dim, thw

        if K == 2:
            xs = xs1
        elif K == 4:
            xs = torch.cat([xs1, xs2], dim=1)
        elif K == 6:
            xs = torch.cat([xs1, xs2, xs3], dim=1)
        elif K == 10:
            xs = torch.cat([xs1, xs2, xs3, xs4, xs5], dim=1)
        elif K == 12:
            xs = torch.cat([xs1, xs2, xs3, xs4, xs5, xs6], dim=1)
        else:
            raise ValueError("K cannot be set as other numbers")

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
        ).view(B, K, -1, L)             # bs, k, dim, h * t * w
        assert out_y.dtype == torch.float

        # out_y = rearrange(out_y, 'b k c (n t) -> (b t) k c n', t=num_frame)
        assert L == H * W * T

        y1 = out_y[:, 0, ...] + torch.flip(out_y[:, 1, ...], dims=[-1]).view(B, -1, L)         # bs, dim, thw   
        y1 = y1.reshape(B, C, T, H, W)          # bs, dim, t, h
        if K > 2:
            y2 = out_y[:, 2, ...] + torch.flip(out_y[:, 3, ...], dims=[-1]).view(B, -1, L)         # bs, dim, twh
            y2 = y2.reshape(B, C, T, W, H).permute(0, 1, 2, 4, 3)            # bs, dim, t, h, w
        if K > 4:
            y3 = out_y[:, 4, ...] + torch.flip(out_y[:, 5, ...], dims=[-1]).view(B, -1, L)         # bs, dim, hwt
            y3 = y3.reshape(B, C, H, W, T).permute(0, 1, 4, 2, 3)            # bs, dim, t, h, w
        if K > 8:
            y4 = out_y[:, 6, ...] + torch.flip(out_y[:, 7, ...], dims=[-1]).view(B, -1, L)         # bs, dim, wht
            y4 = y4.reshape(B, C, W, H, T).permute(0, 1, 4, 3, 2)            # bs, dim, t, h, w
            
            y5 = out_y[:, 8, ...] + torch.flip(out_y[:, 9, ...], dims=[-1]).view(B, -1, L)         # bs, dim, htw
            y5 = y5.reshape(B, C, H, T, W).permute(0, 1, 3, 2, 4)            # bs, dim, t, h, w
            
        if K > 10:
            y6 = out_y[:, 10, ...] + torch.flip(out_y[:, 11, ...], dims=[-1]).view(B, -1, L)         # bs, dim, wth
            y6 = y6.reshape(B, C, W, T, H).permute(0, 1, 3, 4, 2)            # bs, dim, t, h, w
        
        
        if K == 2:
            return y1
        elif K == 4:
            return y1 + y2
        elif K == 6:
            return y1 + y2 + y3
        elif K == 10:
            return y1 + y2 + y3 + y4 + y5
        elif K == 12:   
            return y1 + y2 + y3 + y4 + y5 + y6
  
  
    def forward(self, x: torch.Tensor):
        B, T, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, t, h, w, d)

        if self.use_conv_3d:
            x = x.permute(0, 4, 1, 2, 3).contiguous()       # bs, dim, t, h, w
            x = self.act(self.conv3d(x))             # (b, d, t, h, w)
        else:
            x = rearrange(x, 'b t h w d -> (b t) d h w')         # bs * t, dim, h, w
            x = self.act(self.conv2d(x))
            x = rearrange(x, '(b t) d h w -> b d t h w', t=T)        # bs, dim, t, h, w
        y = self.forward_core(x)
        assert y.dtype == torch.float32        # bs, c, t, h, w
        y = rearrange(y, 'b c t h w -> b t h w c')         # bs, time, h, w, dim
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
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


class Layer(nn.Module):
    def __init__(self,
                 hidden_dim: int = 0,
                 drop_path: float = 0,
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 attn_drop_rate: float = 0,
                 d_state: int = 16,
                 use_3d_conv: bool = True,
                 scan_order: int = 8,
                 **kwargs):
        super().__init__()

        self.ln_l = norm_layer(hidden_dim)
        self.mamba_block = Mamba3D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, use_conv_3d=use_3d_conv, scan_order=scan_order, **kwargs)
        self.drop_path = DropPath(drop_path)
        
    def forward(self, x: torch.Tensor):
        '''
        vis: bs, t, h, w, dim
        audio: bs, t, h, w, dim
        '''
        x = self.drop_path(self.mamba_block(self.ln_l(x))) + x
        
        return x

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


class VMamba_Block(nn.Module):
    def __init__(self,
                d_model,
                d_state=16,
                scan_order=1,
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
                device=None,
                dtype=None,):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.scan_order = scan_order

        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # mamba block内部结构
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
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
    
    def forward_core(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor):
        B, C, H1, W1 = x1.shape
        L1 = H1 * W1
        x_hwwh1 = torch.stack([x1.view(B, C, L1), torch.transpose(x1, dim0=2, dim1=3).contiguous().view(B, C, L1)], dim=1).view(B, 2, C, L1)

        B, C, H2, W2 = x2.shape
        L2 = H2 * W2
        x_hwwh2 = torch.stack([x2.view(B, C, L2), torch.transpose(x2, dim0=2, dim1=3).contiguous().view(B, C, L2)], dim=1).view(B, 2, C, L2)

        B, C, H3, W3 = x3.shape
        L3 = H3 * W3
        x_hwwh3 = torch.stack([x3.view(B, C, L3), torch.transpose(x3, dim0=2, dim1=3).contiguous().view(B, C, L3)], dim=1).view(B, 2, C, L3)
        
        x_hwwh = torch.cat([x_hwwh1, x_hwwh2, x_hwwh3], dim=3)          # bs, 2, dim, l1 + l2 + l3
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)         # bs, 4, dim, l1 + l2 + l3

        L = H1 * W1 + H2 * W2 + H3 * W3
        K = self.K

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, C, L), self.x_proj_weight)
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

        out_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L) + out_y[:, 0:2]       # bs, 2, dim, l
        out_y1 = out_y[:, 0][:, :, :H1 * W1].view(B, -1, H1, W1) + out_y[:, 1][:, :, :H1 * W1].view(B, -1, W1, H1).transpose(2, 3).contiguous()    # bs, dim, h1, w1
        out_y2 = out_y[:, 0][:, :, H1 * W1: (H1 * W1 + H2 * W2)].view(B, -1, H2, W2) +\
              out_y[:, 1][:, :, H1 * W1: (H1 * W1 + H2 * W2)].view(B, -1, W2, H2).transpose(2, 3).contiguous()    # bs, dim, h2, w2
        out_y3 = out_y[:, 0][:, :, (H1 * W1 + H2 * W2): ].view(B, -1, H3, W3) +\
              out_y[:, 1][:, :, (H1 * W1 + H2 * W2): ].view(B, -1, W3, H3).transpose(2, 3).contiguous()    # bs, dim, h3, w3

        return out_y1, out_y2, out_y3
    

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor):
        B, H1, W1, C = x1.shape
        B, H2, W2, C = x2.shape
        B, H3, W3, C = x3.shape

        xz1 = self.in_proj(x1)
        x1, z1 = xz1.chunk(2, dim=-1) # (b, h, w, d)

        xz2 = self.in_proj(x2)
        x2, z2 = xz2.chunk(2, dim=-1)

        xz3 = self.in_proj(x3)
        x3, z3 = xz3.chunk(2, dim=-1)

        x1 = x1.permute(0, 3, 1, 2).contiguous()
        x1 = self.act(self.conv2d(x1)) # (b, d, h, w)

        x2 = x2.permute(0, 3, 1, 2).contiguous()
        x2 = self.act(self.conv2d(x2)) # (b, d, h, w)

        x3 = x3.permute(0, 3, 1, 2).contiguous()
        x3 = self.act(self.conv2d(x3))

        y1, y2, y3 = self.forward_core(x1, x2, x3)      # bs, dim, h, w
        assert y1.dtype == torch.float32
        y1 = rearrange(y1, 'b c h w -> b h w c')
        y2 = rearrange(y2, 'b c h w -> b h w c')
        y3 = rearrange(y3, 'b c h w -> b h w c')

        y1 = self.out_norm(y1)
        y1 = y1 * F.silu(z1)
        out1 = self.out_proj(y1)

        y2 = self.out_norm(y2)
        y2 = y2 * F.silu(z2)
        out2 = self.out_proj(y2)

        y3 = self.out_norm(y3)
        y3 = y3 * F.silu(z3)
        out3 = self.out_proj(y3)

        if self.dropout is not None:
            out1 = self.dropout(out1)
            out2 = self.dropout(out2)
            out3 = self.dropout(out3)

        return out1, out2, out3
    

class TemporalMambaBlock(nn.Module):
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
                device=None,
                dtype=None,
                scan_order=8):
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
    
    def forward_core(self, input_x1: torch.Tensor, input_x2: torch.Tensor, input_x3: torch.Tensor):
        B, C, T, H1, W1 = input_x1.shape
        B, C, T, H2, W2 = input_x2.shape
        B, C, T, H3, W3 = input_x3.shape
        K = self.K

        # 这里先使用8个方向的交互，如果会显著增加训练时间的话就换成6个方向     这里要注意一定是 spatial first thw
        x1 = input_x1.reshape(B, C, -1)       # bs, dim, thw
        x2 = input_x2.reshape(B, C, -1)
        x3 = input_x3.reshape(B, C, -1)
        xs1_ = torch.cat([x1, x2, x3], dim=-1)            # bs, dim, 3 * thw
        xs1 = torch.stack([xs1_, torch.flip(xs1_, dims=[-1])], dim=1)        # bs, 2, dim, 3 * thw

        x1 = input_x1.permute(0, 1, 2, 4, 3).reshape(B, C, -1)       # bs, dim, twh
        x2 = input_x2.permute(0, 1, 2, 4, 3).reshape(B, C, -1)
        x3 = input_x3.permute(0, 1, 2, 4, 3).reshape(B, C, -1)
        xs2_ = torch.cat([x1, x2, x3], dim=-1)            # bs, dim, 3 * twh
        xs2 = torch.stack([xs2_, torch.flip(xs2_, dims=[-1])], dim=1)        # bs, 2, dim, 3 * twh

        x1 = input_x1.permute(0, 1, 3, 4, 2).reshape(B, C, -1)       # bs, dim, hwt
        x2 = input_x2.permute(0, 1, 3, 4, 2).reshape(B, C, -1)
        x3 = input_x3.permute(0, 1, 3, 4, 2).reshape(B, C, -1)
        xs3_ = torch.cat([x1, x2, x3], dim=-1)            # bs, dim, 3 * htw
        xs3 = torch.stack([xs3_, torch.flip(xs3_, dims=[-1])], dim=1)        # bs, 2, dim, 3 * htw

        x1 = input_x1.permute(0, 1, 4, 3, 2).reshape(B, C, -1)       # bs, dim, wht
        x2 = input_x2.permute(0, 1, 4, 3, 2).reshape(B, C, -1)
        x3 = input_x3.permute(0, 1, 4, 3, 2).reshape(B, C, -1)
        xs4_ = torch.cat([x1, x2, x3], dim=-1)            # bs, dim, 3 * wht
        xs4 = torch.stack([xs4_, torch.flip(xs4_, dims=[-1])], dim=1)        # bs, 2, dim, 3 * wht
        
        x1 = input_x1.permute(0, 1, 3, 2, 4).reshape(B, C, -1)       # bs, dim, htw
        x2 = input_x2.permute(0, 1, 3, 2, 4).reshape(B, C, -1)
        x3 = input_x3.permute(0, 1, 3, 2, 4).reshape(B, C, -1)
        xs5_ = torch.cat([x1, x2, x3], dim=-1)            # bs, dim, 3 * htw
        xs5 = torch.stack([xs5_, torch.flip(xs5_, dims=[-1])], dim=1)        # bs, 2, dim, 3 * htw
        
        x1 = input_x1.permute(0, 1, 4, 2, 3).reshape(B, C, -1)       # bs, dim, wth
        x2 = input_x2.permute(0, 1, 4, 2, 3).reshape(B, C, -1)
        x3 = input_x3.permute(0, 1, 4, 2, 3).reshape(B, C, -1)
        xs6_ = torch.cat([x1, x2, x3], dim=-1)            # bs, dim, 3 * wth
        xs6 = torch.stack([xs6_, torch.flip(xs6_, dims=[-1])], dim=1)        # bs, 2, dim, 3 * htw

        # x4 = input_x1.permute(0, 1, 4, 2, 3).reshape(B, C, -1)        # bs, dim, wth
        # xs4 = torch.stack([x4, torch.flip(x4, dims=[-1])], dim=1)        # bs, 2, dim, wth
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

        L = T * (H1 * W1 + H2 * W2 + H3 * W3)
        assert L == xs.shape[-1]

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
        assert L == T * (H1 * W1 + H2 * W2 + H3 * W3)
        L1 = H1 * W1 * T
        L2 = H2 * W2 * T
        L3 = H3 * W3 * T

        y1 = out_y[:, 0, ...] + torch.flip(out_y[:, 1, ...], dims=[-1]).view(B, -1, L)         # bs, dim, 3 * thw 
        out_y1, out_y2, out_y3 = y1[:, :, : L1], y1[:, :, L1: L1 + L2], y1[:, :, L1 + L2:]
        out_y1_ = out_y1.reshape(B, C, T, H1, W1)          # bs, dim, t, h, w
        out_y2_ = out_y2.reshape(B, C, T, H2, W2)          # bs, dim, t, h, w
        out_y3_ = out_y3.reshape(B, C, T, H3, W3)          # bs, dim, t, h, w
        
        if K > 2:
            y2 = out_y[:, 2, ...] + torch.flip(out_y[:, 3, ...], dims=[-1]).view(B, -1, L)         # bs, dim, 3 * twh
            out_y1, out_y2, out_y3 = y2[:, :, : L1], y2[:, :, L1: L1 + L2], y2[:, :, L1 + L2:]
            out_y1__ = out_y1.reshape(B, C, T, W1, H1).permute(0, 1, 2, 4, 3)            # bs, dim, t, h, w
            out_y2__ = out_y2.reshape(B, C, T, W2, H2).permute(0, 1, 2, 4, 3)            # bs, dim, t, h, w
            out_y3__ = out_y3.reshape(B, C, T, W3, H3).permute(0, 1, 2, 4, 3)            # bs, dim, t, h, w
        if K > 4:
            y3 = out_y[:, 4, ...] + torch.flip(out_y[:, 5, ...], dims=[-1]).view(B, -1, L)         # bs, dim, 3 * htw
            out_y1, out_y2, out_y3 = y3[:, :, : L1], y3[:, :, L1: L1 + L2], y3[:, :, L1 + L2:]
            out_y1___ = out_y1.reshape(B, C, H1, W1, T).permute(0, 1, 4, 2, 3)           # bs, dim, t, h, w
            out_y2___ = out_y2.reshape(B, C, H2, W2, T).permute(0, 1, 4, 2, 3)           # bs, dim, t, h, w
            out_y3___ = out_y3.reshape(B, C, H3, W3, T).permute(0, 1, 4, 2, 3)           # bs, dim, t, h, w
        if K > 8:
            y4 = out_y[:, 6, ...] + torch.flip(out_y[:, 7, ...], dims=[-1]).view(B, -1, L)         # bs, dim, 3 * wht
            out_y1, out_y2, out_y3 = y4[:, :, : L1], y4[:, :, L1: L1 + L2], y4[:, :, L1 + L2:]
            out_y1____ = out_y1.reshape(B, C, W1, H1, T).permute(0, 1, 4, 3, 2)           # bs, dim, t, h, w
            out_y2____ = out_y2.reshape(B, C, W2, H2, T).permute(0, 1, 4, 3, 2)           # bs, dim, t, h, w
            out_y3____ = out_y3.reshape(B, C, W3, H3, T).permute(0, 1, 4, 3, 2)           # bs, dim, t, h, w
            
            y5 = out_y[:, 8, ...] + torch.flip(out_y[:, 9, ...], dims=[-1]).view(B, -1, L)         # bs, dim, 3 * wth
            out_y1, out_y2, out_y3 = y5[:, :, : L1], y5[:, :, L1: L1 + L2], y5[:, :, L1 + L2:]
            out_y1_____ = out_y1.reshape(B, C, H1, T, W1).permute(0, 1, 3, 2, 4)           # bs, dim, t, h, w
            out_y2_____ = out_y2.reshape(B, C, H2, T, W2).permute(0, 1, 3, 2, 4)           # bs, dim, t, h, w
            out_y3_____ = out_y3.reshape(B, C, H3, T, W3).permute(0, 1, 3, 2, 4)           # bs, dim, t, h, w
            
        if K > 10:
            y6 = out_y[:, 10, ...] + torch.flip(out_y[:, 11, ...], dims=[-1]).view(B, -1, L)         # bs, dim, 3 * wth
            out_y1, out_y2, out_y3 = y6[:, :, : L1], y6[:, :, L1: L1 + L2], y6[:, :, L1 + L2:]
            out_y1______ = out_y1.reshape(B, C, W1, T, H1).permute(0, 1, 3, 4, 2)           # bs, dim, t, h, w
            out_y2______ = out_y2.reshape(B, C, W2, T, H2).permute(0, 1, 3, 4, 2)           # bs, dim, t, h, w
            out_y3______ = out_y3.reshape(B, C, W3, T, H3).permute(0, 1, 3, 4, 2)           # bs, dim, t, h, w
            
        
        if K == 2:
            y1 = out_y1_ # + out_y1__ # + out_y1___
            y2 = out_y2_ # + out_y2__ # + out_y2___
            y3 = out_y3_ # + out_y3__ # + out_y3___
        elif K == 4:
            y1 = out_y1_ + out_y1__
            y2 = out_y2_ + out_y2__ 
            y3 = out_y3_ + out_y3__ 
        elif K == 6:
            y1 = out_y1_ + out_y1__ + out_y1___
            y2 = out_y2_ + out_y2__ + out_y2___
            y3 = out_y3_ + out_y3__ + out_y3___
        elif K == 10:
            y1 = out_y1_ + out_y1__ + out_y1___ + out_y1____ + out_y1_____
            y2 = out_y2_ + out_y2__ + out_y2___ + out_y2____ + out_y2_____
            y3 = out_y3_ + out_y3__ + out_y3___ + out_y3____ + out_y3_____
        elif K == 12:
            y1 = out_y1_ + out_y1__ + out_y1___ + out_y1____ + out_y1_____ + out_y1______
            y2 = out_y2_ + out_y2__ + out_y2___ + out_y2____ + out_y2_____ + out_y2______
            y3 = out_y3_ + out_y3__ + out_y3___ + out_y3____ + out_y3_____ + out_y3______
        # y4 = out_y[:, 6, ...] + torch.flip(out_y[:, 7, ...], dims=[-1]).view(B, -1, L)         # bs, dim, 3 * htw
        # y4 = rearrange(y4, 'b c (h w t) -> b c w t h', h=H, w=W, t=T).permute(0, 1, 3, 4, 2)            # bs, dim, t, h, w
        
        return y1, y2, y3
        # return y1 + y2 + y3 + y4              # bs, dim, t, h, w
    

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor):
        B, T, H, W, C = x1.shape

        xz1 = self.in_proj(x1)
        x1, z1 = xz1.chunk(2, dim=-1) # (b, t, h, w, d)

        xz2 = self.in_proj(x2)
        x2, z2 = xz2.chunk(2, dim=-1)

        xz3 = self.in_proj(x3)
        x3, z3 = xz3.chunk(2, dim=-1)

        if self.use_conv_3d:
            x1 = x1.permute(0, 4, 1, 2, 3).contiguous()       # bs, dim, t, h, w
            x1 = self.act(self.conv3d(x1))             # (b, d, t, h, w)

            x2 = x2.permute(0, 4, 1, 2, 3).contiguous()       # bs, dim, t, h, w
            x2 = self.act(self.conv3d(x2))             # (b, d, t, h, w)

            x3 = x3.permute(0, 4, 1, 2, 3).contiguous()       # bs, dim, t, h, w
            x3 = self.act(self.conv3d(x3))             # (b, d, t, h, w)
        else:
            x1 = rearrange(x1, 'b t h w d -> (b t) d h w')         # bs * t, dim, h, w
            x1 = self.act(self.conv2d(x1))
            x1 = rearrange(x1, '(b t) d h w -> b d t h w', t=T)        # bs, dim, t, h, w

            x2 = rearrange(x2, 'b t h w d -> (b t) d h w')         # bs * t, dim, h, w
            x2 = self.act(self.conv2d(x2))
            x2 = rearrange(x2, '(b t) d h w -> b d t h w', t=T)        # bs, dim, t, h, w

            x3 = rearrange(x3, 'b t h w d -> (b t) d h w')         # bs * t, dim, h, w
            x3 = self.act(self.conv2d(x3))
            x3 = rearrange(x3, '(b t) d h w -> b d t h w', t=T)        # bs, dim, t, h, w

        y1, y2, y3 = self.forward_core(x1, x2, x3)
        assert y1.dtype == torch.float32        # bs, c, t, h, w
        
        y1 = rearrange(y1, 'b c t h w -> b t h w c')         # bs, time, h, w, dim
        y1 = self.out_norm(y1)
        y1 = y1 * F.silu(z1)
        out1 = self.out_proj(y1)
        if self.dropout is not None:
            out1 = self.dropout(out1)

        y2 = rearrange(y2, 'b c t h w -> b t h w c')         # bs, time, h, w, dim
        y2 = self.out_norm(y2)
        y2 = y2 * F.silu(z2)
        out2 = self.out_proj(y2)
        if self.dropout is not None:
            out2 = self.dropout(out2)

        y3 = rearrange(y3, 'b c t h w -> b t h w c')         # bs, time, h, w, dim
        y3 = self.out_norm(y3)
        y3 = y3 * F.silu(z3)
        out3 = self.out_proj(y3)
        if self.dropout is not None:
            out3 = self.dropout(out3)

        return out1, out2, out3


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
                 mlp_ratio: int = 4,
                 use_temporal_scale: bool = True,
                 use_mix_ffn: bool = False,
                 use_conv_3d: bool = True,
                 scan_order: int = 8,
                 **kwargs,):
        super().__init__()
        '''
        先进行帧内多尺度交互，再进行帧间多尺度特征交互
        '''
        self.ln_l = norm_layer(hidden_dim)
        self.mamba_block = VMamba_Block(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

        self.use_ffn = use_mix_ffn
        if use_mix_ffn:
            self.mixffn = MixFFN(in_features=hidden_dim, 
                                hidden_features=int(mlp_ratio * hidden_dim), 
                                out_features=hidden_dim,
                                )
            self.norm2 = norm_layer(hidden_dim)

        self.use_temporal_scale = use_temporal_scale
        if use_temporal_scale:
            self.norm3 = norm_layer(hidden_dim)
            self.temporal_mamba = TemporalMambaBlock(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, use_conv_3d=use_conv_3d, scan_order=scan_order)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, num_frame: int=5):
        '''
        vis: bs * t, h, w, dim
        audio: bs * t, h, w, dim
        '''
        x1_, x2_, x3_ = self.mamba_block(self.ln_l(x1), self.ln_l(x2), self.ln_l(x3))
        x1 = x1 + self.drop_path(x1_)
        x2 = x2 + self.drop_path(x2_)
        x3 = x3 + self.drop_path(x3_)

        if self.use_ffn:
            x1 = self.drop_path(self.mixffn(self.norm2(x1))) + x1
            x2 = self.drop_path(self.mixffn(self.norm2(x2))) + x2
            x3 = self.drop_path(self.mixffn(self.norm2(x3))) + x3
            
        if self.use_temporal_scale:
            x1 = rearrange(x1, '(b t) h w c -> b t h w c', t=num_frame)
            x2 = rearrange(x2, '(b t) h w c -> b t h w c', t=num_frame)
            x3 = rearrange(x3, '(b t) h w c -> b t h w c', t=num_frame)
            x1_, x2_, x3_ = self.temporal_mamba(self.norm3(x1), self.norm3(x2), self.norm3(x3))
            x1 = x1 + self.drop_path(x1_)
            x2 = x2 + self.drop_path(x2_)
            x3 = x3 + self.drop_path(x3_)
            x1 = rearrange(x1, 'b t h w c -> (b t) h w c', t=num_frame)
            x2 = rearrange(x2, 'b t h w c -> (b t) h w c', t=num_frame)
            x3 = rearrange(x3, 'b t h w c -> (b t) h w c', t=num_frame)

        return x1, x2, x3
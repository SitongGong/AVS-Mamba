import torch
from torch import nn
import math

class Adapter(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.adapter = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
                                     nn.GELU(),
                                     nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1, bias=False), 
                                     nn.GELU())
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') 

# 分别将其作为adapter和lora并检验其效果
class Lora(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=0.25):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.downsample = nn.Linear(embed_dim, hidden_dim)
        self.act_layer = nn.GELU()
        self.upsample = nn.Linear(hidden_dim, embed_dim)

        self.scale = nn.Parameter(torch.ones(1))

        with torch.no_grad():       # initialize the weight and bias
            nn.init.kaiming_uniform_(self.downsample.weight, a=math.sqrt(5))
            nn.init.zeros_(self.upsample.weight)
            nn.init.zeros_(self.downsample.bias)
            nn.init.zeros_(self.upsample.bias)

    def forward(self, x):
        shortcut = x
        x = self.downsample(x)
        x = self.act_layer(x)
        x = self.upsample(x)
        
        return (x * self.scale + shortcut)
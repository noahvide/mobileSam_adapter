import torch.nn as nn

class AdapterModule(nn.Module):
    def __init__(self, dim, bottleneck_dim=64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, dim)
        self.activation = nn.GELU()
    
    def forward(self, x):
        return x + self.up(self.activation(self.down(x)))
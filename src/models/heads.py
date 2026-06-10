import torch.nn as nn
from torch.nn.init import trunc_normal_


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 100):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        trunc_normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.fc(x)


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, num_classes: int = 100, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def build_head(head_type: str, in_dim: int, num_classes: int = 100, **kwargs) -> nn.Module:
    if head_type == "linear":
        return LinearHead(in_dim, num_classes)
    elif head_type == "mlp":
        return MLPHead(in_dim, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown head type: {head_type}")

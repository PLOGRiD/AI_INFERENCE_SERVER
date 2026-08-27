import torch.nn as nn


class SpectralEncoder(nn.Module):
    """18채널 분광(파장 오름차순) -> z_n."""

    def __init__(self, in_dim=18, hidden=(64, 32), z_dim=2, dropout=0.3):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, z_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FusionHead(nn.Module):
    """concat(z_rgb, z_n) -> 2클래스 로짓."""

    def __init__(self, in_dim=4, hidden=(16,), n_cls=2, dropout=0.2):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, n_cls)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

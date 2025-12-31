import torch
import torch.nn as nn

class UNetDown(nn.Module):
    def __init__(self, in_c, out_c, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class UNetUp(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.block(x)
        return torch.cat([x, skip], dim=1)

class GeneratorUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.d1 = UNetDown(1, 64, normalize=False)
        self.d2 = UNetDown(64, 128)
        self.d3 = UNetDown(128, 256)
        self.d4 = UNetDown(256, 512)
        self.d5 = UNetDown(512, 512)
        self.d6 = UNetDown(512, 512)

        self.u1 = UNetUp(512, 512)
        self.u2 = UNetUp(1024, 512)
        self.u3 = UNetUp(1024, 256)
        self.u4 = UNetUp(512, 128)
        self.u5 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 2, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)

        u1 = self.u1(d6, d5)
        u2 = self.u2(u1, d4)
        u3 = self.u3(u2, d3)
        u4 = self.u4(u3, d2)
        u5 = self.u5(u4, d1)

        return self.final(u5)

class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(3, 64, normalize=False),   # 1(L) + 2(ab)
            *block(64, 128),
            *block(128, 256),
            nn.Conv2d(256, 1, 4, padding=1)
        )

    def forward(self, L, ab):
        x = torch.cat([L, ab], dim=1)
        return self.model(x)

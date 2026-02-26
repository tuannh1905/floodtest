"""
Build-up Case 4: + EESP Bottleneck
Delta vs Case 3:
- Bottleneck: thay DoubleConv bằng EESPBottleneck(128, 256, num_blocks=1, branches=4)
- Encoder (TripleBranch), Decoder, Output: giữ nguyên
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.propose_model.module.axial_dw import AxialDW
from models.propose_model.module.dual_vss_block import DualVSSBlock
from models.propose_model.module.eesp_bottleneck import EESPBottleneck


# ── Primitives ────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_c, in_c // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_c // 2 + skip_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ── Triple Branch Encoder (giống Case 3) ──────────────────────
class TripleBranchEncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3):
        super().__init__()

        self.b1_axial = AxialDW(dim=in_c, mixer_kernel=(kernel_size, kernel_size))
        self.b1_pw    = nn.Conv2d(in_c, in_c, kernel_size=1, bias=False)
        self.b1_bn    = nn.BatchNorm2d(in_c)
        self.b1_relu  = nn.ReLU(inplace=True)

        self.b2_axial = AxialDW(dim=in_c, mixer_kernel=(kernel_size, kernel_size))
        self.b2_pw    = nn.Conv2d(in_c, in_c, kernel_size=1, bias=False)
        self.b2_bn    = nn.BatchNorm2d(in_c)
        self.b2_relu  = nn.ReLU(inplace=True)
        self.b2_vss   = DualVSSBlock(hidden_dim=in_c, d_state=8)

        self.b3_avgpool = nn.AdaptiveAvgPool2d(1)
        self.b3_dw      = nn.Conv2d(in_c, in_c, kernel_size=3, padding=1,
                                    groups=in_c, bias=False)
        self.b3_sigmoid = nn.Sigmoid()

        self.channel_adjust = nn.Sequential(
            nn.Conv2d(in_c * 3, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        b1 = self.b1_relu(self.b1_bn(self.b1_pw(self.b1_axial(x))))

        b2 = self.b2_relu(self.b2_bn(self.b2_pw(self.b2_axial(x))))
        b2 = b2.permute(0, 2, 3, 1)
        b2 = self.b2_vss(b2)
        b2 = b2.permute(0, 3, 1, 2)

        b3_attn = self.b3_sigmoid(self.b3_dw(self.b3_avgpool(x)))
        b3      = x * b3_attn

        skip = self.channel_adjust(torch.cat([b1, b2, b3], dim=1))
        x    = self.pool(skip)
        return x, skip


# ── Model ─────────────────────────────────────────────────────
class UNet_EESPBottleneck(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.num_classes = num_classes

        self.stem = nn.Conv2d(in_channels, 16, kernel_size=1, bias=False)

        # Encoder: Triple Branch (giữ nguyên)
        self.e1 = TripleBranchEncoderBlock(16,  16,  kernel_size=3)
        self.e2 = TripleBranchEncoderBlock(16,  32,  kernel_size=3)
        self.e3 = TripleBranchEncoderBlock(32,  64,  kernel_size=3)
        self.e4 = TripleBranchEncoderBlock(64, 128,  kernel_size=3)

        # Bottleneck: NEW -> EESPBottleneck
        self.bottleneck = EESPBottleneck(128, 256, num_blocks=1, branches=4)

        # Decoder: giữ nguyên
        self.d4 = DecoderBlock(256, 128, 128)
        self.d3 = DecoderBlock(128,  64,  64)
        self.d2 = DecoderBlock( 64,  32,  32)
        self.d1 = DecoderBlock( 32,  16,  16)

        # Output: giữ nguyên
        self.out = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)

        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x, skip4 = self.e4(x)

        x = self.bottleneck(x)

        x = self.d4(x,  skip4)
        x = self.d3(x,  skip3)
        x = self.d2(x,  skip2)
        x = self.d1(x,  skip1)

        return self.out(x)


def build_model(num_classes=1):
    return UNet_EESPBottleneck(in_channels=3, num_classes=num_classes)



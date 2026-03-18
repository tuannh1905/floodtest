import torch
import torch.nn as nn
import torch.nn.functional as F

# --- CHIẾN LƯỢC 1: ECAModule (Siêu nhẹ, tăng chất lượng kênh) ---
class ECAModule(nn.Module):
    def __init__(self, channels, k_size=3):
        super(ECAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)

# --- CHIẾN LƯỢC 2: GhostModule (Thay thế Conv 3x3 tiêu chuẩn) ---
class GhostModule(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size=3, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.out_chan = out_chan
        init_chan = out_chan // ratio
        new_chan = init_chan * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_chan, init_chan, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_chan),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_chan, new_chan, dw_size, 1, dw_size//2, groups=init_chan, bias=False),
            nn.BatchNorm2d(new_chan),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_chan, :, :]

# --- KHỐI ĐẶC TRƯNG: DABModule_Lite (Nâng cấp với ECA) ---
class DABModule_Lite(nn.Module):
    def __init__(self, ch, d=1):
        super().__init__()
        # Tích chập bất đối xứng phân tách (Depthwise)
        self.conv3x1 = nn.Conv2d(ch, ch, (3, 1), stride=1, padding=(d, 0), dilation=(d, 1), groups=ch, bias=False)
        self.conv1x3 = nn.Conv2d(ch, ch, (1, 3), stride=1, padding=(0, d), dilation=(1, d), groups=ch, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU(inplace=True)
        self.eca = ECAModule(ch) # Bù nhãn quan

    def forward(self, x):
        out = self.conv3x1(x)
        out = self.conv1x3(out)
        out = self.bn(out)
        out = self.relu(out + x)
        return self.eca(out)

# --- CHIẾN LƯỢC 2: DownSamplingBlock_Lite (Ghost-hóa) ---
class DownSamplingBlock_Lite(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.main_branch = GhostModule(in_chan, out_chan - in_chan, stride=2)
        self.pool_branch = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = torch.cat([self.main_branch(x), self.pool_branch(x)], dim=1)
        return out

# --- KIẾN TRÚC TỔNG THỂ: DABNet_FloodLite ---
class DABNet_FloodLite(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        # Initial Block
        self.initial = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Stage 1 (Low level)
        self.down1 = DownSamplingBlock_Lite(32, 64)
        self.stage1 = nn.Sequential(
            DABModule_Lite(64, d=2),
            DABModule_Lite(64, d=2),
            DABModule_Lite(64, d=2)
        )

        # Stage 2 (Mid-High level) - ÉP KÊNH CHIẾN LƯỢC 1 (128 -> 80)
        self.down2 = DownSamplingBlock_Lite(64, 80)
        self.stage2 = nn.Sequential(
            DABModule_Lite(80, d=4),
            DABModule_Lite(80, d=4),
            DABModule_Lite(80, d=8),
            DABModule_Lite(80, d=8),
            DABModule_Lite(80, d=16),
            DABModule_Lite(80, d=16)
        )

        # --- CHIẾN LƯỢC 3: Classifier Head (Bottleneck) ---
        # Tổng hợp feature: 3 (input) + 64 (stage1) + 80 (stage2) = 147 channels
        self.classifier = nn.Sequential(
            nn.Conv2d(147, 64, 1, bias=False), # Bottleneck ép từ 147 xuống 64
            nn.BatchNorm2d(64),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(64, num_classes, 1) # Final output
        )

    def forward(self, x):
        size = x.size()[2:]
        # Nhánh input thu nhỏ để concat cuối
        x_down = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True)
        
        feat_initial = self.initial(x)
        feat_s1 = self.down1(feat_initial)
        feat_s1 = self.stage1(feat_s1)
        
        feat_s2 = self.down2(feat_s1)
        feat_s2 = self.stage2(feat_s2)

        # Concat Feature (DABNet Style)
        # S1 (64) + S2 (80) + X_down (3) = 147
        feat_s1_down = F.interpolate(feat_s1, scale_factor=0.5, mode='bilinear', align_corners=True)
        out = torch.cat([feat_s1_down, feat_s2, x_down], dim=1)
        
        out = self.classifier(out)
        return F.interpolate(out, size, mode='bilinear', align_corners=True)

# Kiểm tra thực tế
if __name__ == '__main__':
    model = DABNet_FloodLite(num_classes=1)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ DABNet-FloodLite Params: {total_params / 1e6:.3f}M")
    # Dự kiến: ~0.45M - 0.48M (Đạt mục tiêu < 0.5M)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# 1. COMPONENT STRATEGY (CÁC KHỐI TOÁN HỌC "GIÁ RẺ")
# =========================================================================

# [ĐÃ SỬA]: Ghost Module thay thế cho Conv 1x1 (Tiết kiệm 40% tham số)
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.PReLU(init_channels) if relu else nn.Identity(), # [ĐÃ SỬA]: Dùng PReLU
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.PReLU(new_channels) if relu else nn.Identity(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


# [ĐÃ SỬA]: Axial Depthwise Conv thay cho Conv 3x3 thông thường (Tiết kiệm 33% tham số)
class AxialDWConv(nn.Module):
    def __init__(self, channels, stride=1, dilation=1):
        super(AxialDWConv, self).__init__()
        self.conv = nn.Sequential(
            # Nhánh 1x3
            nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=(1, stride), 
                      padding=(0, dilation), dilation=(1, dilation), groups=channels, bias=False),
            # Nhánh 3x1
            nn.Conv2d(channels, channels, kernel_size=(3, 1), stride=(stride, 1), 
                      padding=(dilation, 0), dilation=(dilation, 1), groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels)
        )
    def forward(self, x):
        return self.conv(x)


# =========================================================================
# 2. CONTEXT STRATEGY (BẮT BỐI CẢNH)
# =========================================================================

# [ĐÃ SỬA]: Strip Pooling thay cho Global Average Pooling để bắt dải nước lũ
class StripPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((1, None))  # Dải ngang
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))  # Dải dọc
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        
        self.out_conv = GhostModule(in_channels + out_channels, out_channels, relu=True)

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), mode='bilinear', align_corners=False)
        x2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), mode='bilinear', align_corners=False)
        
        sp_feat = torch.sigmoid(x1 + x2) * x
        return self.out_conv(torch.cat([x, sp_feat], dim=1))


# =========================================================================
# 3. STRUCTURAL STRATEGY (CẤU TRÚC RÚT GỌN 3 NHÁNH & THIN DECODER)
# =========================================================================

class SlimDownsamplerBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(SlimDownsamplerBlock, self).__init__()
        self.ghost = GhostModule(in_planes, out_planes, relu=False)
        self.axial = AxialDWConv(out_planes, stride=stride, dilation=1)

    def forward(self, input):
        return self.axial(self.ghost(input))


# [ĐÃ SỬA]: Giảm từ 5 nhánh xuống 3 nhánh (AvgPool, d=1, d=6)
class SlimDilatedParallelConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(SlimDilatedParallelConvBlock, self).__init__()
        # Bắt buộc out_planes chia hết cho 3 để chia đều cho 3 nhánh
        assert out_planes % 3 == 0 
        inter_planes = out_planes // 3
        
        self.ghost_down = GhostModule(in_planes, inter_planes, relu=True)
        
        # 3 Nhánh tính toán song song
        self.pool = nn.AvgPool2d(3, stride=stride, padding=1)
        self.conv1 = AxialDWConv(inter_planes, stride=stride, dilation=1)
        self.conv2 = AxialDWConv(inter_planes, stride=stride, dilation=6) # ASPP rút gọn
        
        self.attention = nn.Conv2d(out_planes, 3, 1, padding=0, bias=False)
        self.ghost_fuse = GhostModule(out_planes, out_planes, relu=False)
        self.act = nn.PReLU(out_planes)

    def forward(self, input):
        out = self.ghost_down(input)
        
        p = self.pool(out)
        d1 = self.conv1(out)
        d2 = self.conv2(out)
        
        # Liên kết chuỗi (Cascade)
        d1 = d1 + p
        d2 = d1 + d2
        
        # Attention
        cat_feat = torch.cat([p, d1, d2], 1)
        att = torch.sigmoid(self.attention(cat_feat))
        
        p = p * att[:, 0].unsqueeze(1)
        d1 = d1 * att[:, 1].unsqueeze(1)
        d2 = d2 * att[:, 2].unsqueeze(1)
        
        output = self.ghost_fuse(torch.cat([p, d1, d2], 1))
        # Residual connection
        if input.shape == output.shape:
            output = output + input
            
        return self.act(output)


# =========================================================================
# 4. MAIN NETWORK: MINISEG SLIM
# =========================================================================
class MiniSeg_Slim(nn.Module):
    def __init__(self, num_classes=1, P1=2, P2=3, P3=8, P4=6):
        super(MiniSeg_Slim, self).__init__()
        
        # [ĐÃ SỬA]: Asymmetric Width Scaling (Đảm bảo chia hết cho 3)
        # C1 (L2D): ~ alpha = 0.5 -> 12 channels
        # C2, C3, C4 (GFE): ~ alpha = 0.35 -> 24, 36, 48 channels
        C1, C2, C3, C4 = 12, 24, 36, 48
        
        # --- ENCODER ---
        self.stem = SlimDownsamplerBlock(3, C1, stride=2)
        self.level1 = nn.Sequential(*[SlimDilatedParallelConvBlock(C1, C1) for _ in range(P1)])
        
        self.down2 = SlimDownsamplerBlock(C1, C2, stride=2)
        self.level2 = nn.Sequential(*[SlimDilatedParallelConvBlock(C2, C2) for _ in range(P2)])
        
        self.down3 = SlimDownsamplerBlock(C2, C3, stride=2)
        self.level3 = nn.Sequential(*[SlimDilatedParallelConvBlock(C3, C3) for _ in range(P3)])
        
        self.down4 = SlimDownsamplerBlock(C3, C4, stride=2)
        self.level4 = nn.Sequential(*[SlimDilatedParallelConvBlock(C4, C4) for _ in range(P4)])

        # --- CONTEXT ---
        self.strip_pool = StripPooling(C4, C4)

        # --- THIN DECODER (Backbone-less concept) ---
        # [ĐÃ SỬA]: Chỉ dùng F.interpolate và GhostModule cực mỏng để khôi phục
        self.dec_up3 = GhostModule(C4 + C3, C3, relu=True)
        self.dec_up2 = GhostModule(C3 + C2, C2, relu=True)
        self.dec_up1 = GhostModule(C2 + C1, C1, relu=True)
        
        self.pred = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(C1, num_classes, 1)
        )

    def forward(self, input):
        # Lược bỏ hoàn toàn cơ chế split() và cross-concat phức tạp của bản gốc
        # để tránh phân mảnh bộ nhớ (memory fragmentation) trên thiết bị nhúng.
        
        l1 = self.level1(self.stem(input))
        l2 = self.level2(self.down2(l1))
        l3 = self.level3(self.down3(l2))
        l4 = self.level4(self.down4(l3))
        
        l4 = self.strip_pool(l4)
        
        # Decoding
        up3 = F.interpolate(l4, size=l3.size()[2:], mode='bilinear', align_corners=False)
        d3 = self.dec_up3(torch.cat([up3, l3], 1))
        
        up2 = F.interpolate(d3, size=l2.size()[2:], mode='bilinear', align_corners=False)
        d2 = self.dec_up2(torch.cat([up2, l2], 1))
        
        up1 = F.interpolate(d2, size=l1.size()[2:], mode='bilinear', align_corners=False)
        d1 = self.dec_up1(torch.cat([up1, l1], 1))
        
        out = self.pred(d1)
        return F.interpolate(out, input.size()[2:], mode='bilinear', align_corners=False)


def build_model(num_classes=1):
    return MiniSeg_Slim(num_classes=num_classes)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =========================================================================
# 1. ECA MODULE (Attention Kênh)
# =========================================================================
class ECA_Module(nn.Module):
    def __init__(self, channels, b=1, gamma=2):
        super(ECA_Module, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

# =========================================================================
# 2. CÁC LỚP BỔ TRỢ & DSCONV (Chìa khóa hạ tham số)
# =========================================================================
class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)
        return output

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()
        self.bn_acti = bn_acti
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)
        if self.bn_acti:
            output = self.bn_prelu(output)
        return output

class DSConv(nn.Module):
    """Depthwise Separable Convolution: Tiết kiệm ~8 lần tham số so với Conv chuẩn"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# =========================================================================
# 3. DAB MODULE (Cốt lõi DABNet)
# =========================================================================
class DABModule(nn.Module):
    def __init__(self, nIn, d=1, kSize=3, dkSize=3):
        super().__init__()
        self.bn_relu_1 = BNPReLU(nIn)
        
        # Đã DSConv-hóa để ép tham số xuống < 0.3M
        self.conv3x3 = nn.Sequential(
            DSConv(nIn, nIn // 2, kernel_size=kSize, stride=1, padding=1),
            BNPReLU(nIn // 2)
        )

        self.dconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1, 0), groups=nIn // 2, bn_acti=True)
        self.dconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)

        self.bn_relu_2 = BNPReLU(nIn // 2)
        self.conv1x1 = Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)
        
        self.eca = ECA_Module(nIn)

    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv3x3(output)

        br1 = self.dconv1x3(self.dconv3x1(output))
        br2 = self.ddconv1x3(self.ddconv3x1(output))

        output = br1 + br2
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)

        output = self.eca(output)
        return output + input

class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut
        nConv = nOut - nIn if self.nIn < self.nOut else nOut

        self.dsconv = DSConv(nIn, nConv, kernel_size=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.dsconv(input)
        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)
        return self.bn_prelu(output)

class InputInjection(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList([nn.AvgPool2d(3, stride=2, padding=1) for _ in range(ratio)])

    def forward(self, input):
        for pool in self.pool: 
            input = pool(input)
        return input

# =========================================================================
# 4. BLOCK LITE-ASPP ATTENTION (Giá siêu rẻ: ~25k tham số)
# =========================================================================
class LiteASPPAttention(nn.Module):
    def __init__(self, ch_in, ch_pool=16, num_classes=1):
        super(LiteASPPAttention, self).__init__()
        
        # 1. Các nhánh Pooling đa tầm nhìn (Parameter-free)
        self.gap_pool = nn.AdaptiveAvgPool2d(1)
        self.aspp_pool = nn.AdaptiveAvgPool2d(2)
        
        # Conv ép kênh (Siêu nhẹ)
        self.gap_conv = Conv(ch_in, ch_pool, kernel_size=1, stride=1, padding=0, bn_acti=True)
        self.aspp_conv = Conv(ch_in, ch_pool, kernel_size=1, stride=1, padding=0, bn_acti=True)
        
        # Nhánh gốc (Bảo toàn chi tiết rìa nước)
        self.detail_conv = Conv(ch_in, ch_pool, kernel_size=1, stride=1, padding=0, bn_acti=True)
        
        # 2. Classifier Head (Fusion)
        # 3 nhánh * 16 channels = 48 channels
        concat_channels = 3 * ch_pool 
        
        self.classifier = nn.Sequential(
            nn.Conv2d(concat_channels, concat_channels, kernel_size=3, padding=1, groups=concat_channels, bias=False), # Depthwise
            nn.Conv2d(concat_channels, 64, kernel_size=1, bias=False), # Pointwise
            nn.BatchNorm2d(64),
            nn.Hardswish(inplace=True), 
            nn.Dropout2d(0.1),
            nn.Conv2d(64, num_classes, kernel_size=1, bias=True)
        )
        
    def forward(self, x):
        size = x.size()[2:] 
        
        # Nhánh 1: Global Context (Trị lỗi quy mô lớn)
        out_gap = self.gap_pool(x)
        out_gap = self.gap_conv(out_gap)
        out_gap = F.interpolate(out_gap, size=size, mode='bilinear', align_corners=False)
        
        # Nhánh 2: Spatial Context (Kết nối vệt nước)
        out_aspp = self.aspp_pool(x)
        out_aspp = self.aspp_conv(out_aspp)
        out_aspp = F.interpolate(out_aspp, size=size, mode='bilinear', align_corners=False)
        
        # Nhánh 3: Local Detail 
        out_detail = self.detail_conv(x)
        
        # Fusion 48 kênh
        feat_fusion = torch.cat([out_detail, out_gap, out_aspp], dim=1)
        
        # Đưa ra kết quả cuối
        out = self.classifier(feat_fusion)
        return out

# =========================================================================
# 5. MẠNG CHÍNH (LIGHT DABNET + LITE-ASPP V7)
# =========================================================================
class LightDABNet(nn.Module):
    def __init__(self, classes=1, block_1=3, block_2=6, ch_b1=64, ch_b2=96):
        super().__init__()
        
        # Init conv đã được DSConv hóa để ép tham số
        self.init_conv = nn.Sequential(
            Conv(3, 32, 3, 2, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True, groups=32), # Depthwise
            Conv(32, 32, 1, 1, padding=0, bn_acti=True),            # Pointwise
            Conv(32, 32, 3, 1, padding=1, bn_acti=True, groups=32), # Depthwise
            Conv(32, 32, 1, 1, padding=0, bn_acti=True),            # Pointwise
        )

        self.down_1 = InputInjection(1)  
        self.down_2 = InputInjection(2)  
        self.down_3 = InputInjection(3)  

        self.bn_prelu_1 = BNPReLU(32 + 3)

        # Stage 1
        self.downsample_1 = DownSamplingBlock(32 + 3, ch_b1)
        self.DAB_Block_1 = nn.Sequential()
        for i in range(0, block_1):
            self.DAB_Block_1.add_module("DAB_Module_1_" + str(i), DABModule(ch_b1, d=2))
        self.bn_prelu_2 = BNPReLU((ch_b1 * 2) + 3)

        # Stage 2 (Hybrid Dilation - Tránh Gridding)
        dilation_block_2 = [1, 2, 4, 8, 1, 2] 
        in_ch_b2 = (ch_b1 * 2) + 3
        self.downsample_2 = DownSamplingBlock(in_ch_b2, ch_b2)
        self.DAB_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            self.DAB_Block_2.add_module("DAB_Module_2_" + str(i), DABModule(ch_b2, d=dilation_block_2[i]))
        self.bn_prelu_3 = BNPReLU((ch_b2 * 2) + 3)

        # Classifier Đa Tầm Nhìn (Lite-ASPP)
        concat_channels = (ch_b2 * 2) + 3 # 195 channels
        self.aspp_classifier = LiteASPPAttention(ch_in=concat_channels, ch_pool=16, num_classes=classes)

    def forward(self, input):
        output0 = self.init_conv(input)

        down_1 = self.down_1(input)
        down_2 = self.down_2(input)
        down_3 = self.down_3(input)

        output0_cat = self.bn_prelu_1(torch.cat([output0, down_1], 1))

        output1_0 = self.downsample_1(output0_cat)
        output1 = self.DAB_Block_1(output1_0)
        output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, down_2], 1))

        output2_0 = self.downsample_2(output1_cat)
        output2 = self.DAB_Block_2(output2_0)
        output2_cat = self.bn_prelu_3(torch.cat([output2, output2_0, down_3], 1))

        # Phân loại
        out = self.aspp_classifier(output2_cat)
        
        # Phóng to về kích thước gốc
        out = F.interpolate(out, size=input.size()[2:], mode='bilinear', align_corners=False)

        return out

# =========================================================================
# HÀM BUILD MODEL CHUẨN TEMPLATE
# =========================================================================
def build_model(num_classes=1):
    """
    Khởi tạo mạng LightDABNet V7 siêu tối ưu toàn diện:
    1. DSConv Cốt lõi + Init -> Ép tham số về ngưỡng ~130k-150k.
    2. Hybrid Dilation [1, 2, 4, 8, 1, 2] -> Bảo toàn chi tiết đô thị.
    3. LiteASPP Classifier (+25k) -> Cung cấp bối cảnh toàn cục.
    """
    return LightDABNet(classes=num_classes, block_1=3, block_2=6, ch_b1=64, ch_b2=96)
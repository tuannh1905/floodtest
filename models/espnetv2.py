import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

# --- Helper Layers for ESPNetv2 ---

class ConvBNPReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNPReLU, self).__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(out_ch)
        )

class EESP(nn.Module):
    '''Khối EESP - Linh hồn của ESPNetv2'''
    def __init__(self, in_ch, out_ch, stride=1, k=4, r_lim=7):
        super(EESP, self).__init__()
        self.stride = stride
        n = out_ch // k
        n_remainder = out_ch % k
        self.proj_1x1 = ConvBNPReLU(in_ch, n, kernel_size=1, groups=k)
        
        self.s_pyramid = nn.ModuleList()
        for i in range(k):
            dilation = min(2**i, r_lim)
            self.s_pyramid.append(nn.Conv2d(n, n, 3, stride, dilation, dilation, groups=n, bias=False))
        
        self.merge = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.PReLU(out_ch)
        )
        self.pw_1x1 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, 1, groups=k, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.act = nn.PReLU(out_ch)

    def forward(self, x):
        out_1x1 = self.proj_1x1(x)
        outputs = [layer(out_1x1) for layer in self.s_pyramid]
        
        # Merge các nhánh dilation
        merged = outputs[0]
        for i in range(1, len(outputs)):
            merged = torch.cat([merged, outputs[i]], 1)
        
        out_merge = self.merge(merged)
        out_pw = self.pw_1x1(out_merge)
        
        if self.stride == 1 and x.shape == out_pw.shape:
            out_pw = out_pw + x
            
        return self.act(out_pw)

class EfficientPyrPool(nn.Module):
    '''Pyramid Pooling cực nhẹ của ESPNetv2'''
    def __init__(self, in_planes, out_planes, last_layer_br=True):
        super().__init__()
        self.main_path = nn.Conv2d(in_planes, out_planes, 3, 1, 1, groups=out_planes, bias=False)
        self.group_pw = nn.Conv2d(out_planes, out_planes, 1, 1, groups=4, bias=False)
        self.bn_act = nn.Sequential(
            nn.BatchNorm2d(out_planes),
            nn.PReLU(out_planes)
        ) if last_layer_br else nn.Identity()

    def forward(self, x):
        x = self.main_path(x)
        x = self.group_pw(x)
        return self.bn_act(x)

# --- Main Model ---

class ESPNetv2(nn.Module):
    def __init__(self, num_classes=1, scale=1.0):
        super().__init__()
        # Cấu hình kênh dựa trên scale (s=1.0 là mặc định)
        config = [int(x * scale) for x in [32, 64, 128, 256, 512]]
        
        # Encoder
        self.level1 = ConvBNPReLU(3, config[0], 3, 2)
        self.level2 = EESP(config[0], config[1], stride=2)
        self.level3 = nn.Sequential(
            EESP(config[1], config[2], stride=2),
            EESP(config[2], config[2], stride=1)
        )
        self.level4 = nn.Sequential(
            EESP(config[2], config[3], stride=2),
            EESP(config[3], config[3], stride=1)
        )
        
        # Decoder (Simplified Bottom-up)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Các lớp merge Feature từ Encoder sang Decoder
        self.proj_l3 = nn.Conv2d(config[2], config[1], 1, bias=False)
        self.proj_l2 = nn.Conv2d(config[1], config[0], 1, bias=False)
        
        # Head cuối
        self.classifier = nn.Sequential(
            EfficientPyrPool(config[0], config[0]),
            nn.Conv2d(config[0], num_classes, 1)
        )
        
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        size = x.size()[2:]
        
        # Encoder
        c1 = self.level1(x)    # 1/2
        c2 = self.level2(c1)   # 1/4
        c3 = self.level3(c2)   # 1/8
        c4 = self.level4(c3)   # 1/16
        
        # Decoder
        d3 = self.up(c4)       # 1/8
        d3 = self.up(d3)       # 1/4 (Nhảy bậc để demo, chuẩn ESPNetv2 decode dần)
        
        # Kết quả cuối up lên size gốc
        out = self.classifier(c1) # Dùng feature sớm để giữ chi tiết
        return F.interpolate(out, size=size, mode='bilinear', align_corners=True)

def build_model(num_classes=1):
    '''
    Mặc định dùng scale s=1.0 cho cân bằng giữa tốc độ và độ chính xác.
    '''
    return ESPNetv2(num_classes=num_classes, scale=1.0)
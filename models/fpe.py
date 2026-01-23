import torch
import torch.nn as nn
import torch.nn.functional as F


class FPEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride, dilations=[1, 2, 4, 8]):
        super(FPEBlock, self).__init__()
        assert len(dilations) > 0, 'Length of dilations should be larger than 0'
        
        self.K = len(dilations)
        self.use_skip = (in_channels == out_channels) and (stride == 1)
        expand_channels = out_channels * expansion
        self.ch = expand_channels // self.K
        
        # Initial 1x1 convolution
        self.conv_init = nn.Conv2d(in_channels, expand_channels, kernel_size=1, bias=False)
        self.bn_init = nn.BatchNorm2d(expand_channels)
        self.relu_init = nn.ReLU(inplace=True)
        
        # Depthwise convolutions with different dilations
        self.dw_convs = nn.ModuleList()
        self.dw_bns = nn.ModuleList()
        self.dw_relus = nn.ModuleList()
        for i in range(self.K):
            dilation = dilations[i]
            padding = dilation
            self.dw_convs.append(nn.Conv2d(self.ch, self.ch, kernel_size=3, stride=stride, 
                                           padding=padding, dilation=dilation, groups=self.ch, bias=False))
            self.dw_bns.append(nn.BatchNorm2d(self.ch))
            self.dw_relus.append(nn.ReLU(inplace=True))
        
        # Final 1x1 convolution
        self.conv_last = nn.Conv2d(expand_channels, out_channels, kernel_size=1, bias=False)
        self.bn_last = nn.BatchNorm2d(out_channels)
        self.relu_last = nn.ReLU(inplace=True)
    
    def forward(self, x):
        if self.use_skip:
            residual = x
        
        # Initial convolution
        x = self.conv_init(x)
        x = self.bn_init(x)
        x = self.relu_init(x)
        
        # Process each channel group with different dilations
        transform_feats = []
        for i in range(self.K):
            feat = x[:, i*self.ch:(i+1)*self.ch]
            feat = self.dw_convs[i](feat)
            feat = self.dw_bns[i](feat)
            feat = self.dw_relus[i](feat)
            transform_feats.append(feat)
        
        # Accumulate features
        for j in range(1, self.K):
            transform_feats[j] = transform_feats[j] + transform_feats[j-1]
        
        # Concatenate all features
        x = torch.cat(transform_feats, dim=1)
        
        # Final convolution
        x = self.conv_last(x)
        x = self.bn_last(x)
        x = self.relu_last(x)
        
        # Skip connection
        if self.use_skip:
            x = x + residual
        
        return x


class SpatialAttentionBlock(nn.Module):
    def __init__(self):
        super(SpatialAttentionBlock, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelAttentionBlock(nn.Module):
    def __init__(self, channels):
        super(ChannelAttentionBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MEUModule(nn.Module):
    def __init__(self, low_channels, high_channels, out_channels):
        super(MEUModule, self).__init__()
        # Low-level feature processing
        self.conv_low = nn.Conv2d(low_channels, out_channels, kernel_size=1, bias=False)
        self.bn_low = nn.BatchNorm2d(out_channels)
        self.relu_low = nn.ReLU(inplace=True)
        
        # High-level feature processing
        self.conv_high = nn.Conv2d(high_channels, out_channels, kernel_size=1, bias=False)
        self.bn_high = nn.BatchNorm2d(out_channels)
        self.relu_high = nn.ReLU(inplace=True)
        
        # Attention modules
        self.sa = SpatialAttentionBlock()
        self.ca = ChannelAttentionBlock(out_channels)
    
    def forward(self, x_low, x_high):
        # Process low-level features
        x_low = self.conv_low(x_low)
        x_low = self.bn_low(x_low)
        x_low = self.relu_low(x_low)
        
        # Process high-level features
        x_high = self.conv_high(x_high)
        x_high = self.bn_high(x_high)
        x_high = self.relu_high(x_high)
        
        # Apply attention
        x_sa = self.sa(x_low)
        x_ca = self.ca(x_high)
        
        # Fuse features
        x_low = x_low * x_ca
        x_high = F.interpolate(x_high, scale_factor=2, mode='bilinear', align_corners=True)
        x_high = x_high * x_sa
        
        return x_low + x_high


class FPENetModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, p=3, q=9, k=4):
        """
        FPENet model for real-time semantic segmentation
        
        Args:
            in_channels: Number of input channels (default: 3 for RGB)
            num_classes: Number of output classes
            p: Number of FPE blocks in stage 2
            q: Number of FPE blocks in stage 3
            k: Expansion factor for FPE blocks
        """
        super(FPENetModel, self).__init__()
        
        self.num_classes = num_classes
        
        # Stage 1
        self.stage1_conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.stage1_bn = nn.BatchNorm2d(16)
        self.stage1_relu = nn.ReLU(inplace=True)
        self.stage1_fpe = FPEBlock(16, 16, 1, 1)
        
        # Stage 2
        self.stage2_0 = FPEBlock(16, 32, k, 2)
        self.stage2_blocks = nn.ModuleList()
        for i in range(p - 1):
            self.stage2_blocks.append(FPEBlock(32, 32, k, 1))
        
        # Stage 3
        self.stage3_0 = FPEBlock(32, 64, k, 2)
        self.stage3_blocks = nn.ModuleList()
        for i in range(q - 1):
            self.stage3_blocks.append(FPEBlock(64, 64, k, 1))
        
        # Decoder
        self.decoder2 = MEUModule(32, 64, 64)
        self.decoder1 = MEUModule(16, 64, 32)
        
        # Final output
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1, bias=False)
        self.final_bn = nn.BatchNorm2d(num_classes)
        self.final_relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        size = x.size()[2:]
        
        # Stage 1
        x = self.stage1_conv(x)
        x = self.stage1_bn(x)
        x = self.stage1_relu(x)
        x1 = self.stage1_fpe(x)
        
        # Stage 2
        x = self.stage2_0(x1)
        for block in self.stage2_blocks:
            x = block(x)
        x2 = x
        
        # Stage 3
        x = self.stage3_0(x2)
        for block in self.stage3_blocks:
            x = block(x)
        
        # Decoder
        x = self.decoder2(x2, x)
        x = self.decoder1(x1, x)
        
        # Final output
        x = self.final_conv(x)
        x = self.final_bn(x)
        x = self.final_relu(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        
        return x


def build_model(num_classes=1):
    return FPENetModel(in_channels=3, num_classes=num_classes)
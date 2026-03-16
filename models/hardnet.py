import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, bias=False):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel,          
                                         stride=stride, padding=kernel//2, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(out_channels))
        self.add_module('relu', nn.ReLU6(True))                                          

class DWConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super().__init__()
        self.add_module('dwconv', nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                          stride=stride, padding=1, groups=in_channels, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(in_channels))

class CombConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, bias=False):
        super().__init__()
        self.add_module('layer1', ConvLayer(in_channels, out_channels, kernel))
        self.add_module('layer2', DWConvLayer(out_channels, out_channels, stride=stride))

class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, dwconv=False):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0
        for i in range(n_layers):
            outch, inch, link = self.get_link(i+1, in_channels, growth_rate, grmul)
            self.links.append(link)
            if dwconv:
                layers_.append(CombConvLayer(inch, outch))
            else:
                layers_.append(ConvLayer(inch, outch))
            
            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch
        self.layers = nn.ModuleList(layers_)
        
    def forward(self, x):
        layers_ = [x]
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = [layers_[i] for i in link]
            x = torch.cat(tin, 1) if len(tin) > 1 else tin[0]
            out = self.layers[layer](x)
            layers_.append(out)
            
        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keepBase) or (i == t-1) or (i % 2 == 1):
                out_.append(layers_[i])
        return torch.cat(out_, 1)

class HarDNet(nn.Module):
    def __init__(self, num_classes=1, arch=39, depth_wise=True):
        super().__init__()
        # Default settings for HarDNet39DS (Lightweight version)
        first_ch = [24, 48]
        grmul = 1.6
        gr = [16, 20, 64, 160]
        n_layers = [4, 16, 8, 4]
        downSamp = [1, 1, 1, 0]
        ch_list = [96, 320, 640, 1024]

        if arch == 68:
            first_ch = [32, 64]
            grmul = 1.7
            gr = [14, 16, 20, 40, 160]
            n_layers = [8, 16, 16, 16, 4]
            downSamp = [1, 0, 1, 1, 0]
            ch_list = [128, 256, 320, 640, 1024]

        self.encoder = nn.ModuleList([])
        # First layers
        self.encoder.append(ConvLayer(3, first_ch[0], kernel=3, stride=2))
        self.encoder.append(ConvLayer(first_ch[0], first_ch[1], kernel=3))
        self.encoder.append(DWConvLayer(first_ch[1], first_ch[1], stride=2) if depth_wise else nn.MaxPool2d(3, stride=2, padding=1))

        # HarDNet Blocks
        ch = first_ch[1]
        for i in range(len(n_layers)):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i], dwconv=depth_wise)
            ch = blk.out_channels
            self.encoder.append(blk)
            self.encoder.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]
            if downSamp[i] == 1:
                self.encoder.append(DWConvLayer(ch, ch, stride=2) if depth_wise else nn.MaxPool2d(2, stride=2))

        # Segmentation Head (Decoder thay thế cho phần Classification)
        self.head = nn.Sequential(
            nn.Conv2d(ch, num_classes, kernel_size=1),
            # HarDNet39DS downsamples 8x hoặc 16x, ta upsample lại đúng size gốc
        )

    def forward(self, x):
        size = x.size()[2:]
        for layer in self.encoder:
            x = layer(x)
        x = self.head(x)
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

def build_model(num_classes=1):
    # Trả về HarDNet39DS - phiên bản cực nhẹ và nhanh cho mobile/embedded
    return HarDNet(num_classes=num_classes, arch=39, depth_wise=True)
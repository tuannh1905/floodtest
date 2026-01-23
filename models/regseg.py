import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_channels, out_channels, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, 
                    padding=0, bias=bias)


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                    bias=False, act_type='relu', **kwargs):
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            padding = ((kernel_size[0] - 1) // 2 * dilation, (kernel_size[1] - 1) // 2 * dilation)
        elif isinstance(kernel_size, int):    
            padding = (kernel_size - 1) // 2 * dilation

        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
            nn.BatchNorm2d(out_channels),
            Activation(act_type, **kwargs)
        )


class Activation(nn.Module):
    def __init__(self, act_type, **kwargs):
        super().__init__()
        activation_hub = {'relu': nn.ReLU, 'relu6': nn.ReLU6,
                          'leakyrelu': nn.LeakyReLU, 'prelu': nn.PReLU,
                          'celu': nn.CELU, 'elu': nn.ELU, 
                          'hardswish': nn.Hardswish, 'hardtanh': nn.Hardtanh,
                          'gelu': nn.GELU, 'glu': nn.GLU, 
                          'selu': nn.SELU, 'silu': nn.SiLU,
                          'sigmoid': nn.Sigmoid, 'softmax': nn.Softmax, 
                          'tanh': nn.Tanh, 'none': nn.Identity,
                        }

        act_type = act_type.lower()
        if act_type not in activation_hub.keys():
            raise NotImplementedError(f'Unsupport activation type: {act_type}')

        self.activation = activation_hub[act_type](**kwargs)

    def forward(self, x):
        return self.activation(x)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction_ratio, act_type):
        super().__init__()
        squeeze_channels = int(channels * reduction_ratio)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.se_block = nn.Sequential(
                            nn.Linear(channels, squeeze_channels),
                            Activation(act_type),
                            nn.Linear(squeeze_channels, channels),
                            Activation('sigmoid')
                        )

    def forward(self, x):
        residual = x
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.se_block(x).unsqueeze(-1).unsqueeze(-1)
        x = x * residual
        return x


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, r1=None, r2=None, 
                    g=16, se_ratio=0.25, act_type='relu'):
        super().__init__()
        assert stride in [1, 2], f'Unsupported stride: {stride}'
        self.stride = stride

        self.conv1 = ConvBNAct(in_channels, out_channels, 1, act_type=act_type)
        if stride == 1:
            assert in_channels == out_channels, 'In_channels should be the same as out_channels when stride = 1'
            split_ch = out_channels // 2
            assert split_ch % g == 0, 'Group width `g` should be evenly divided by split_ch'
            groups = split_ch // g
            self.split_channels = split_ch
            self.conv_left = ConvBNAct(split_ch, split_ch, 3, dilation=r1, groups=groups, act_type=act_type)
            self.conv_right = ConvBNAct(split_ch, split_ch, 3, dilation=r2, groups=groups, act_type=act_type)
        else:
            assert out_channels % g == 0, 'Group width `g` should be evenly divided by out_channels'
            groups = out_channels // g
            self.conv_left = ConvBNAct(out_channels, out_channels, 3, 2, groups=groups, act_type=act_type)
            self.conv_skip = nn.Sequential(
                                nn.AvgPool2d(2, 2, 0),
                                ConvBNAct(in_channels, out_channels, 1, act_type='none')
                            )
        self.conv2 = nn.Sequential(
                        SEBlock(out_channels, se_ratio, act_type),
                        ConvBNAct(out_channels, out_channels, 1, act_type='none')
                    )
        self.act = Activation(act_type)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        if self.stride == 1:
            x_left = self.conv_left(x[:, :self.split_channels])
            x_right = self.conv_right(x[:,self.split_channels:])
            x = torch.cat([x_left, x_right], dim=1)
        else:
            x = self.conv_left(x)
            residual = self.conv_skip(residual)

        x = self.conv2(x)
        x += residual
        return self.act(x)


class Decoder(nn.Module):
    def __init__(self, num_class, d4_channel, d8_channel, d16_channel, act_type):
        super().__init__()
        self.conv_d16 = ConvBNAct(d16_channel, 128, 1, act_type=act_type)
        self.conv_d8_stage1 = ConvBNAct(d8_channel, 128, 1, act_type=act_type)
        self.conv_d4_stage1 = ConvBNAct(d4_channel, 8, 1, act_type=act_type)
        self.conv_d8_stage2 = ConvBNAct(128, 64, 3, act_type=act_type)
        self.conv_d4_stage2 = nn.Sequential(
                                    ConvBNAct(64+8, 64, 3, act_type=act_type),
                                    conv1x1(64, num_class)
                                )

    def forward(self, x_d4, x_d8, x_d16):
        size_d4 = x_d4.size()[2:]
        size_d8 = x_d8.size()[2:]

        x_d16 = self.conv_d16(x_d16)
        x_d16 = F.interpolate(x_d16, size_d8, mode='bilinear', align_corners=True)

        x_d8 = self.conv_d8_stage1(x_d8)
        x_d8 += x_d16
        x_d8 = self.conv_d8_stage2(x_d8)
        x_d8 = F.interpolate(x_d8, size_d4, mode='bilinear', align_corners=True)

        x_d4 = self.conv_d4_stage1(x_d4)
        x_d4 = torch.cat([x_d4, x_d8], dim=1)
        x_d4 = self.conv_d4_stage2(x_d4)
        return x_d4


class RegSegModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, dilations=None, act_type='relu'):
        super(RegSegModel, self).__init__()
        if dilations is None:
            dilations = [[1,1], [1,2], [1,2], [1,3], [2,3], [2,7], [2,3],
                         [2,6], [2,5], [2,9], [2,11], [4,7], [5,14]]
        else:
            if len(dilations) != 13:
                raise ValueError("Dilation pairs' length should be 13\n")

        self.num_classes = num_classes
        self.conv_init = ConvBNAct(in_channels, 32, 3, 2, act_type=act_type)

        self.stage_d4 = DBlock(32, 48, 2, act_type=act_type)

        layers = [DBlock(48, 128, 2, act_type=act_type)]
        for _ in range(3-1):
            layers.append(DBlock(128, 128, 1, r1=1, r2=1, act_type=act_type))
        self.stage_d8 = nn.Sequential(*layers)

        layers = [DBlock(128, 256, 2, act_type=act_type)]
        for i in range(13-1):
            layers.append(DBlock(256, 256, 1, r1=dilations[i][0], r2=dilations[i][1], act_type=act_type))

        layers.append(DBlock(256, 320, 2, r1=dilations[-1][0], r2=dilations[-1][1], act_type=act_type))
        self.stage_d16 = nn.Sequential(*layers)

        self.decoder = Decoder(num_classes, 48, 128, 320, act_type)

    def forward(self, x):
        size = x.size()[2:]

        x = self.conv_init(x)
        x_d4 = self.stage_d4(x)
        x_d8 = self.stage_d8(x_d4)
        x_d16 = self.stage_d16(x_d8)
        x = self.decoder(x_d4, x_d8, x_d16)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return x


def build_model(num_classes=1):
    return RegSegModel(in_channels=3, num_classes=num_classes)
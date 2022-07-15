from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(Conv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, channels=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.down_conv1 = Conv(in_channels, channels[0])
        self.down_conv2 = Conv(channels[0], channels[1])
        self.down_conv3 = Conv(channels[1], channels[2])
        self.down_conv4 = Conv(channels[2], channels[3])
        self.down_sample = nn.MaxPool2d(2, 2)

        self.bottleneck = Conv(channels[3], channels[3])

        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.up_conv1 = Conv(2 * channels[3], channels[2])
        self.up_conv2 = Conv(2 * channels[2], channels[1])
        self.up_conv3 = Conv(2 * channels[1], channels[0])
        self.up_conv4 = Conv(2 * channels[0], channels[0])

        self.out_conv = nn.Conv2d(channels[0], out_channels, 3, 1, 1)

    def forward(self, x):
        # B, 1, 565, 584
        x1 = self.down_conv1(x)  # B, 64, 565, 584
        x2 = self.down_conv2(self.down_sample(x1))  # B, 128, 282, 292
        x3 = self.down_conv3(self.down_sample(x2))  # B, 256, 141, 146
        x4 = self.down_conv4(self.down_sample(x3))  # B, 512, 70, 73
        y = self.bottleneck(self.down_sample(x4))  # B, 512, 35, 36
        y1 = self.up_sample(y)  # B, 512, 70, 72
        y1 = F.pad(y1, (x4.shape[3] - y1.shape[3], 0, x4.shape[2] - y1.shape[2], 0))
        y2 = self.up_sample(self.up_conv1(torch.cat((x4, y1), dim=1)))
        y2 = F.pad(y2, (x3.shape[3] - y2.shape[3], 0, x3.shape[2] - y2.shape[2], 0))
        y3 = self.up_sample(self.up_conv2(torch.cat((x3, y2), dim=1)))
        y3 = F.pad(y3, (x2.shape[3] - y3.shape[3], 0, x2.shape[2] - y3.shape[2], 0))
        y4 = self.up_sample(self.up_conv3(torch.cat((x2, y3), dim=1)))
        y4 = F.pad(y4, (x1.shape[3] - y4.shape[3], 0, x1.shape[2] - y4.shape[2], 0))
        y5 = self.up_conv4(torch.cat((x1, y4), dim=1))
        out = self.out_conv(y5)
        return {'out': out}


from typing import Dict


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class UNet_(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet_, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return {"out": logits}
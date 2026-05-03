import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Final 1x1 convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)       # 572 -> 568
        p1 = self.pool1(e1)     # 568 -> 284

        e2 = self.enc2(p1)      # 284 -> 280
        p2 = self.pool2(e2)     # 280 -> 140

        e3 = self.enc3(p2)      # 140 -> 136
        p3 = self.pool3(e3)     # 136 -> 68

        e4 = self.enc4(p3)      # 68 -> 64
        p4 = self.pool4(e4)     # 64 -> 32

        # Bottleneck
        b = self.bottleneck(p4) # 32 -> 28

        # Decoder
        d4 = self.up4(b)        # 28 -> 56
        e4 = self.crop_to_match(e4, d4)
        d4 = torch.cat([e4, d4], dim=1)  # 512 + 512 = 1024
        d4 = self.dec4(d4)      # 56 -> 52

        d3 = self.up3(d4)       # 52 -> 104
        e3 = self.crop_to_match(e3, d3)
        d3 = torch.cat([e3, d3], dim=1)  # 256 + 256 = 512
        d3 = self.dec3(d3)      # 104 -> 100

        d2 = self.up2(d3)       # 100 -> 200
        e2 = self.crop_to_match(e2, d2)
        d2 = torch.cat([e2, d2], dim=1)  # 128 + 128 = 256
        d2 = self.dec2(d2)      # 200 -> 196

        d1 = self.up1(d2)       # 196 -> 392
        e1 = self.crop_to_match(e1, d1)
        d1 = torch.cat([e1, d1], dim=1)  # 64 + 64 = 128
        d1 = self.dec1(d1)      # 392 -> 388

        logits = self.final_conv(d1)     # 388 -> 388

        return logits

    def crop_to_match(self, encoder_feature, decoder_feature):
        _, _, h, w = decoder_feature.shape
        _, _, H, W = encoder_feature.shape

        top = (H - h) // 2
        left = (W - w) // 2

        return encoder_feature[:, :, top:top + h, left:left + w]
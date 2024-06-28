import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, n_class, depth, start_filters):
        super().__init__()
        self.depth = depth
        
        # Define layers for encoding
        self.encoders = nn.ModuleList()
        self.pooling = nn.ModuleList()

        in_channels = 1
        out_channels = start_filters
        
        for _ in range(depth):
            self.encoders.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
            self.pooling.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
            out_channels *= 2

        # Bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Define layers for decoding
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for _ in range(depth):
            self.upconvs.append(
                nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=2, stride=2)
            )
            self.decoders.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
            out_channels //= 2

        # Output layer
        self.outconv = nn.Conv2d(out_channels, n_class, kernel_size=1)

    def encode(self, x, encoder, pool):
        x = encoder(x)
        pooled = pool(x)
        return x, pooled

    def decode(self, x, skip, upconv, decoder):
        x = upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = decoder(x)
        return x

    def forward(self, x):
        skips = []
        for encoder, pool in zip(self.encoders, self.pooling):
            x, pooled = self.encode(x, encoder, pool)
            skips.append(x)
            x = pooled

        x = self.bottleneck(x)

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = self.decode(x, skip, upconv, decoder)

        out = self.outconv(x)
        return out

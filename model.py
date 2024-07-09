import torch
import torch.nn as nn
import torch.nn.functional as F

"""
UNet model for image segmentation

The model consists of a downward path and an upward path. 
The downward path consists of a series of convolutions and pooling layers that reduce the spatial dimensions of the input image. 
The upward path consists of a series of transposed convolutions and concatenations with the skip connections from the downward path.

For more information and references to use cases, refer to:
https://paperswithcode.com/method/u-net
"""
class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(Unet, self).__init__()
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downward path
        for feature in features:
            self.down.append(nn.Sequential(
                nn.Conv2d(in_channels, feature, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(0.5)
            ))
            in_channels = feature

        # Upward path
        for feature in reversed(features):
            self.up.append(nn.Sequential(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2),
                nn.Conv2d(feature * 2, feature, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(0.5)
            ))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1] * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []

        for down in self.down:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for idx, up in enumerate(self.up):
            x = up[0](x)  # Transposed convolution
            x = F.interpolate(x, size=skips[idx].shape[2:], mode='bilinear', align_corners=True) # Upsample
            x = torch.cat((x, skips[idx]), dim=1) # Concatenate skip connection
            x = up[1:](x)  # Convolutions after concatenation

        return self.final_conv(x)

# Example of using the UNet model
if __name__ == "__main__":
    model = Unet()
    x = torch.randn(1, 3, 128, 256)  # Example input size
    out = model(x)
    print(out.shape)  # Expected output size
"""
Discriminator model for D2BGAN (Low Light Image Enhancement)

Programmed by Sasika Amarasinghe <sasikapamith2016@gmail.com>
* 2024-03-12: Initial coding
"""
from common_blocks import ConvBlock, ResidualBlock, TransposeConvBlock
from torch import nn
import torch

class Discriminator(nn.Module):
  def __init__(self, in_channels=3):
    super().__init__()

    self.discriminator = nn.Sequential(
        ConvBlock(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1, stride=1, is_relu=True),
        ConvBlock(in_channels=64, out_channels=64*2, kernel_size=3, padding=1, stride=1, is_relu=True),
        ConvBlock(in_channels=64*2, out_channels=64*4, kernel_size=3, padding=1, stride=1, is_relu=True),
        ConvBlock(in_channels=64*4, out_channels=64*8, kernel_size=3, padding=1, stride=1, is_relu=True),

        ConvBlock(in_channels=64*8, out_channels=1, kernel_size=3, padding=1, stride=1, is_relu=True)

    )

  def forward(self,x):
    return self.discriminator(x)

  def debug(self,x):
    for i, layer in enumerate(self.discriminator):
      print(i)

def test_discriminator():
    images_per_batch = 5
    img_channels = 3
    img_size = 128
    x = torch.randn((images_per_batch, img_channels, img_size, img_size))
    disc = Discriminator(in_channels=img_channels)
    print("input : ", x.shape)
    print("discriminator output : " , disc(x).shape)
    # gen.debug(x)


if __name__ == "__main__":
    test_discriminator()

"""
Generator model for CycleGAN

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
* 2020-11-05: Initial coding
* 2022-12-21: Small revision of code, checked that it works with latest PyTorch version
"""

from common_blocks import ConvBlock, ResidualBlock, TransposeConvBlock
from torch import nn
import torch 

class GeneratorEncoder(nn.Module):
  def __init__(self, in_channels=3, out_channels=64):
    super().__init__()

    self.encoder = nn.Sequential(
        ConvBlock(in_channels, out_channels, kernel_size=3, padding=1, stride=1, is_relu=True),
        ConvBlock(out_channels, out_channels*2, kernel_size=3, padding=1, stride=1, is_relu=True),
        ConvBlock(out_channels*2, out_channels*4, kernel_size=3, padding=1, stride=1, is_relu=True),

        ResidualBlock(out_channels*4),
        ResidualBlock(out_channels*4)
    )


  def forward(self, x):
    x = self.encoder(x)

    return x

class GeneratorDecoder(nn.Module):
  def __init__(self, in_channels=3, out_channels=64):
    super().__init__()

    self.decoder = nn.Sequential(
        ResidualBlock(out_channels*4, padding_mode="zeros"),
        ResidualBlock(out_channels*4, padding_mode="zeros"),

        TransposeConvBlock(out_channels*4, out_channels*2,kernel_size=3,padding=1, stride=1, is_relu=True, padding_mode="zeros"),
        TransposeConvBlock(out_channels*2, out_channels,kernel_size=3,padding=1, stride=1, is_relu=True, padding_mode="zeros"),
        TransposeConvBlock(out_channels, out_channels=in_channels,kernel_size=3,padding=1, stride=1, is_relu=True, padding_mode="zeros"),

    )


  def forward(self, x):
    x = self.decoder(x)

    return x

# Tests

def test_generator_encoder():
    images_per_batch = 5
    img_channels = 3
    img_size = 128
    x = torch.randn((images_per_batch, img_channels, img_size, img_size))
    gen_encoder = GeneratorEncoder(in_channels=3,out_channels=64)
    print("input : ", x.shape)
    print("encoder of generator output : " , gen_encoder(x).shape)

def test_generator_decoder():
    images_per_batch = 5
    img_channels = 3
    img_size = 128
    x = torch.randn((images_per_batch, 256, img_size, img_size))
    gen_decoder = GeneratorDecoder()
    print("input : ", x.shape)
    print("decoder of generator output : " , gen_decoder(x).shape)

class Generator(nn.Module):
  def __init__(self, in_channels=3, out_channels=64):
    super().__init__()

    self.encoder = GeneratorEncoder(in_channels=in_channels, out_channels=out_channels)
    self.decoder = GeneratorDecoder(in_channels=in_channels, out_channels=out_channels)

  def forward(self,x):
    x = self.encoder(x)
    x = self.decoder(x)

    return x

  def debug(self,x):
    print("Input : " ,x.shape)
    x = self.encoder(x)
    print("After encoder : ", x.shape)
    x = self.decoder(x)
    print("After decoder : ", x.shape)

    return x

def test_generator():
  images_per_batch = 5
  img_channels = 4
  img_size = 128
  num_features = 50  # num of features in the feature vector (number of channels in an image in the encoder output)

  x = torch.randn((images_per_batch, img_channels, img_size, img_size))
  gen = Generator(in_channels=img_channels, out_channels=num_features)
  output = gen(x)
  gen.debug(x)
  print("generator output :", output.shape)


if __name__ == "__main__":
    test_generator_encoder()
    test_generator_decoder()
    test_generator()

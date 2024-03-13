import torch
import torch.nn as nn

###################################################
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, padding, stride, padding_mode="reflect", is_relu=True, down=True):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, padding_mode=padding_mode),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if is_relu else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)
    
    
def test_ConvBlock():
    images_per_batch = 5
    img_channels = 3
    img_size = 128
    x = torch.randn((images_per_batch, img_channels, img_size, img_size))
    conv_block = ConvBlock(in_channels=img_channels, out_channels=256, kernel_size=3,padding=1, stride=1, is_relu=True)
    print("input : ", x.shape)
    print("decoder of generator output : " , conv_block(x).shape)
    

########################################################
class ResidualBlock(nn.Module):
  def __init__(self, channels, padding_mode="reflect"): # Because of Residual block height, width or number of channels will not be changed.
    super().__init__()

    self.block = nn.Sequential(
        # There are two conv blocks in a residual block
        ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, stride=1, is_relu=False, padding_mode=padding_mode),  # padding 1 , stride = 1 for no change in height and width
        ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, stride=1, is_relu=False, padding_mode=padding_mode)
    )

  def forward(self, x):
    return x + self.block(x)

  def print_architecture(self):
    print(self)

  def debug_ResidualBlock(self,x):
    print("Input : ", x.shape)

    # Iterate through conv blocks
    for i, block in enumerate(self.block):
      print(i)
      for j, layer in enumerate(self.block[i].conv):
        print("\t\t",j, layer)

        x = layer(x)
        # print()
        print(x.shape)



def test_ResidualBlock():
    images_per_batch = 5
    img_channels = 3
    out_channels = 56
    img_size = 128
    x = torch.randn((images_per_batch, 256, img_size, img_size))
    res_block = ResidualBlock(channels=256)
    # gen.print_architecture()
    # res_block.debug_ResidualBlock(x)
    print("input : ", x.shape)
    print("decoder of generator output : " , res_block(x).shape)

########################################################
class TransposeConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels,kernel_size,padding, stride, padding_mode="reflect", is_relu=True, down=True):
    super().__init__()

    self.transposeConv = nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, padding_mode=padding_mode),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(inplace=True) if is_relu else nn.Identity()
    )

  def forward(self, x):
    return self.transposeConv(x)


def test():
    test_ConvBlock()
    test_ResidualBlock()
    
if __name__ == "__main__":
    test()
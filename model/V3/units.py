import torch
from torch import nn

class Conv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride = 1,padding = 0,dilation = 1,act = "leaky_relu"):
        super(Conv,self).__init__()

        if act == "identity":
            act_func = nn.Identity()
        elif act == "relu":
            act_func = nn.ReLU()
        elif act == "leaky_relu":
            act_func = nn.LeakyReLU(0.1015625)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size = kernel_size,stride = stride,padding = padding,dilation = dilation,bias = False),
            nn.BatchNorm2d(out_channels),
            act_func
        )

    def forward(self,x):
        return self.conv(x)
class ResBlock(nn.Module):
    def __init__(self,in_channels,act = "leaky_relu"):
        super(ResBlock,self).__init__()
        assert in_channels % 2 == 0, "in_channels must be even for ResBlock"
        self.conv1 = Conv(in_channels = in_channels,out_channels= in_channels // 2, kernel_size = 1, padding = 0, act = act)
        self.conv2 = Conv(in_channels//2,in_channels,kernel_size = 3,padding = 1, act = act)
    
    def forward(self,x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out

if __name__ == "__main__":

    block = ResBlock(in_channels = 64, act = "leaky_relu")
    x = torch.randn((1,64,416,416))
    output = block(x)
    print(output.shape)
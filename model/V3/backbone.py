from units import Conv,ResBlock
import torch
import torch.nn as nn
class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53,self).__init__()
        self.conv1 = Conv(in_channels=3, out_channels=32, kernel_size=3,stride = 1, padding = 1, act = "leaky_relu")
        self.res_block1 = self.build_conv_and_resblock(in_channels = 32, num_blocks = 1)
        self.res_block2 = self.build_conv_and_resblock(in_channels = 64,num_blocks = 2)
        self.res_block3 = self.build_conv_and_resblock(in_channels = 128, num_blocks = 8)
        self.res_block4 = self.build_conv_and_resblock(in_channels = 256,num_blocks = 8)
        self.res_block5 = self.build_conv_and_resblock(in_channels = 512, num_blocks= 4)


    def build_conv_and_resblock(self, in_channels,num_blocks):
        model = nn.Sequential()
        model.add_module("conv",Conv(in_channels = in_channels, out_channels = in_channels * 2,kernel_size = 3, stride = 2, padding=1,act = "leaky_relu"))
        for idx in range(num_blocks):
            model.add_module(f"res{idx}", ResBlock(in_channels = in_channels * 2))
        
        return model
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.res_block1(out)
        out = self.res_block2(out)
        C3 = self.res_block3(out)
        C4 = self.res_block4(C3)
        C5 = self.res_block5(C4)

        return C3,C4,C5
    

def build_darknet():
    feat_dims = (256,512,1024)
    darknet = Darknet53()
    return darknet,feat_dims

if __name__ == "__main__":
    #Example
    x = torch.randn(size = (1,3,416,416))
    darknet,feat_dims = build_darknet()
    out = darknet(x)
    for preds in out:
        print(preds.shape)
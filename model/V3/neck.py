import torch
import torch.nn as nn
from units import Conv

class TopDownLayer(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv1 = Conv(in_channels = in_channels,out_channels = out_channels,kernel_size = 1,stride = 1,padding = 0,act = "leaky_relu")
        self.conv2 = Conv(in_channels = out_channels, out_channels = out_channels*2,kernel_size=3,stride = 1,padding = 1,act = "leaky_relu")
        self.conv3 = Conv(in_channels = out_channels*2, out_channels = out_channels, kernel_size = 1,stride = 1,padding = 0,act = "leaky_relu")
        self.conv4 = Conv(in_channels = out_channels,out_channels = out_channels * 2,kernel_size = 3, stride = 1,padding = 1,act = "leaky_relu")
        self.conv5 = Conv(in_channels = out_channels * 2,out_channels = out_channels,kernel_size = 1,stride = 1,padding = 0,act = "leaky_relu")

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        return out
    
class FPN(nn.Module):
    def __init__(self,feat_dims):
        super().__init__()
        self.feat_dims = feat_dims
        self.topdown1  = TopDownLayer(in_channels = self.feat_dims[2],out_channels=self.feat_dims[-1] // 2)
        self.topdown2 = TopDownLayer(in_channels = self.feat_dims[1] + self.feat_dims[2]//4,out_channels = self.feat_dims[-2]//2)
        self.topdown3 = TopDownLayer(in_channels = self.feat_dims[0] + self.feat_dims[1]//4,out_channels = self.feat_dims[-3]//2)
        self.conv1 = Conv(in_channels = self.feat_dims[2]//2,out_channels = self.feat_dims[2]//4,kernel_size = 1,stride = 1,padding = 0,act = "leaky_relu")
        self.conv2 = Conv(in_channels = self.feat_dims[1]//2,out_channels = self.feat_dims[1]//4,kernel_size = 1,stride = 1,padding = 0,act = "leaky_relu")
        self.upsample = nn.Upsample(scale_factor=2)


    def forward(self,x):
        ftr_s,ftr_m,ftr_l = x
        C1= self.topdown1(ftr_l)
        P1 = self.upsample(self.conv1(C1))
        C2 = self.topdown2(torch.cat((P1,ftr_m),dim = 1))
        P2 = self.upsample(self.conv2(C2))
        C3 = self.topdown3(torch.cat((P2,ftr_s),dim = 1))
        return C1,C2,C3



if __name__ == "__main__":
    from backbone import build_darknet
    input_size = 416
    device = torch.device('cpu')
    backbone,feat_dims = build_darknet()
    neck = FPN(feat_dims = feat_dims)

    x = torch.randn(1,3,input_size,input_size).to(device)
    ftrs = backbone(x)
    ftrs = neck(ftrs)

    for preds in ftrs:
        print(preds.shape)


    # C1 = neck.topdownlayer1(ftrs[2])
    # P1 = neck.upsample(neck.conv1(C1))
    # print("--------------")
    # print(C1.shape, P1.shape)
    # C2_temp = torch.cat((ftrs[1],P1),dim = 1)
    # print("----------------")
    # print(C2_temp.shape)
    # print("-------------")
    # C2 = neck.topdownlayer2(C2_temp)
    # print(C2.shape)
    # print("-------------------------")
    # P2 = neck.upsample(neck.conv2(C2))
    # print(P2.shape)

    # C3_temp = torch.cat((ftrs[0],P2),dim = 1)
    # C3 = neck.topdownlayer3(C3_temp)
    # print("-------------")
    # print(C3.shape)
    # print(neck.upsample(neck.conv1()))
    # ftrs = neck(ftrs)
    # for ftrs in ftrs:
    #     print(ftrs.shape)

   
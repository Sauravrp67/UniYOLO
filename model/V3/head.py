from units import Conv
import torch
import torch.nn as nn
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import set_grid

class Detection_Layer(nn.Module):
    def __init__(self,in_channels, input_size, stride, num_classes,anchors):
        super().__init__()
        self.stride = stride
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = len(self.anchors)
        self.num_attributes = 1 + 4 + self.num_classes
        self.input_size = input_size


        self.conv = Conv(in_channels = in_channels // 2, out_channels = in_channels,kernel_size = 3,stride = 1,padding = 1,act = "leaky_relu")
        self.detect  = nn.Conv2d(in_channels = in_channels, out_channels = self.num_anchors * self.num_attributes,kernel_size = 1,padding =0)
        self.set_grid_xy(input_size = input_size)

    def forward(self,x):
        self.device = x.device

        bs = x.shape[0]

        out = self.conv(x)
        out = self.detect(out)
        out = out.permute(0,2,3,1).flatten(1,2).view((bs,-1,self.num_anchors,self.num_attributes))
        pred_obj = torch.sigmoid(out[...,:1])
        pred_box_txty = torch.sigmoid(out[...,1:3])
        pred_box_twth = torch.sigmoid(out[...,3:5])
        pred_cls = out[...,5:]

        if self.training:
            return torch.cat((pred_obj,pred_box_txty,pred_box_twth,pred_cls),dim = -1)
        else:
            # ### Process the txty and twth values to grid values in normalized form so that converting it to original image size will be easy. (Just multiply the normalized coordinates by image size)
            preds_boxes = self.transform_pred_boxes(torch.cat((pred_box_txty,pred_box_twth),dim = -1))
            pred_scores = pred_obj * torch.sigmoid(pred_cls) 
            pred_score, pred_label = pred_scores.max(dim=-1)
            pred_out = torch.cat((pred_score.unsqueeze(-1),preds_boxes,pred_label.unsqueeze(-1)),dim = -1)
            return pred_out.flatten(1,2)
            # return torch.cat((pred_box_txty,pred_box_twth),dim = -1)
        #.view((bs,self.input_size//self.stride,self.input_size // self.stride,self.num_anchors,4))
    
    def set_grid_xy(self,input_size):
        self.grid_size = input_size // self.stride
        grid_x,grid_y = set_grid(grid_size = self.grid_size)
        self.grid_x = grid_x.contiguous().view((1, -1, 1))
        self.grid_y = grid_y.contiguous().view((1, -1, 1))
    
    def transform_pred_boxes(self,pred_box):
        xc = (pred_box[...,0] + self.grid_x.to(self.device)) / self.grid_size
        yc = (pred_box[...,1] + self.grid_y.to(self.device)) / self.grid_size
        w  = torch.exp(pred_box[...,2]) * self.anchors[:,0].to(self.device)
        h = torch.exp(pred_box[...,3]) * self.anchors[:,1].to(self.device)
        return torch.stack((xc,yc,w,h),dim = -1)
    

    def set_grid_xy(self,input_size):
        self.grid_size = input_size // self.stride
        grid_x,grid_y = set_grid(grid_size = self.grid_size)
        self.grid_x = grid_x.contiguous().view((1, -1, 1))
        self.grid_y = grid_y.contiguous().view((1, -1, 1))

class YoloHead(nn.Module):
    def __init__(self,in_channels, input_size, num_classes,anchors):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.anchors = torch.tensor(anchors) if not torch.is_tensor(anchors) else anchors

        self.detect_l = Detection_Layer(in_channels = in_channels[-1],input_size = 416,stride = 32,num_classes = self.num_classes,anchors = self.anchors[6:9])
        self.detect_m = Detection_Layer(in_channels = in_channels[-2],input_size = 416,stride = 16,num_classes = self.num_classes,anchors = self.anchors[3:6])
        self.detect_s = Detection_Layer(in_channels = in_channels[-3],input_size = 416,stride = 8,num_classes=self.num_classes,anchors = self.anchors[0:3])
        
    def forward(self,x):
        C1,C2,C3 = x
        pred_l = self.detect_l(C1)
        pred_m = self.detect_m(C2)
        pred_s = self.detect_s(C3)

        return pred_s,pred_m,pred_l
    
if __name__ == "__main__":
    from backbone import build_darknet
    from neck import FPN
    input_size = 416
    num_classes =80
    anchors = [[0.248,      0.7237237 ],
        [0.36144578, 0.53      ],
        [0.42,       0.9306667 ],
        [0.456,      0.6858006 ],
        [0.488,      0.8168168 ],
        [0.6636637,  0.274     ],
        [0.806,      0.648     ],
        [0.8605263,  0.8736842 ],
        [0.944,      0.5733333 ]]
    x = torch.randn(size = (8,3,416,416))

    darknet,feat_dims = build_darknet()
    # print(feat_dims[-1])
    neck = FPN(feat_dims = feat_dims)
    head = YoloHead(in_channels=feat_dims,input_size = input_size,num_classes = num_classes,anchors = anchors)
    ftrs = darknet(x)
    ftrs1 = neck(ftrs)
    ftrs = head(ftrs1)

    head.eval()
    ftrs_eval = head(ftrs1)
    for ftrs in ftrs_eval:
        print(ftrs.shape)
    print(torch.cat(ftrs_eval,dim = 1).view(8,-1,6).shape)
    
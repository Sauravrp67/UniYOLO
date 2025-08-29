import sys
from pathlib import Path

import torch
from torch import nn

from units import Conv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import set_grid


class DetectLayer(nn.Module):
    def __init__(self, input_size, in_channels, num_classes, anchors, stride):
        super().__init__()
        self.stride = stride
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_attributes = 1 + 4 + num_classes
        self.conv = Conv(in_channels, in_channels*2, kernel_size=3, padding=1, act="leaky_relu")
        self.detect = nn.Conv2d(in_channels*2, self.num_attributes*self.num_anchors, kernel_size=1, padding=0)
        self.set_grid_xy(input_size=input_size)


    def forward(self, x):
        self.device = x.device
        bs = x.shape[0]

        out = self.conv(x)
        out = self.detect(out)
        out = out.permute(0, 2, 3, 1).flatten(1, 2).view((bs, -1, self.num_anchors, self.num_attributes))

        pred_obj = torch.sigmoid(out[..., :1])
        pred_box_txty = torch.sigmoid(out[..., 1:3])
        pred_box_twth = out[..., 3:5]
        pred_cls = out[..., 5:]

        if self.training:
            return torch.cat((pred_obj, pred_box_txty, pred_box_twth, pred_cls), dim=-1)
        else:
            pred_box = self.transform_pred_box(torch.cat((pred_box_txty, pred_box_twth), dim=-1))
            pred_score = pred_obj * torch.sigmoid(pred_cls)
            pred_score, pred_label = pred_score.max(dim=-1)
            pred_out = torch.cat((pred_score.unsqueeze(-1), pred_box, pred_label.unsqueeze(-1)), dim=-1)
            return pred_out.flatten(1, 2)


    def transform_pred_box(self, pred_box):
        xc = (pred_box[..., 0] + self.grid_x.to(self.device)) / self.grid_size
        yc = (pred_box[..., 1] + self.grid_y.to(self.device)) / self.grid_size
        w = torch.exp(pred_box[..., 2]) * self.anchors[:, 0].to(self.device)
        h = torch.exp(pred_box[..., 3]) * self.anchors[:, 1].to(self.device)
        return torch.cat((xc.unsqueeze(-1), yc.unsqueeze(-1), w.unsqueeze(-1), h.unsqueeze(-1)), dim=-1)


    def set_grid_xy(self, input_size):
        self.grid_size = input_size // self.stride
        grid_x, grid_y = set_grid(grid_size=self.grid_size)
        self.grid_x = grid_x.contiguous().view((1, -1, 1))
        self.grid_y = grid_y.contiguous().view((1, -1, 1))

class DetectLayer_DPU(nn.Module):
    def __init__(self, input_size, in_channels, num_classes, anchors, stride):
        super().__init__()
        self.stride = stride
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_attributes = 1 + 4 + num_classes
        self.conv = Conv(in_channels, in_channels*2, kernel_size=3, padding=1, act="leaky_relu")
        self.detect = nn.Conv2d(in_channels*2, self.num_attributes*self.num_anchors, kernel_size=1, padding=0)
        self.num_classes = num_classes
        self.grid_x,self.grid_y = self.set_grid_xy(input_size=input_size)
        # self.replicate = nn.Conv2d(1,self.num_classes,kernel_size = 1,bias = False)
        # with torch.no_grad():
        #     self.replicate.weight.fill_(1.0)
        #     self.replicate.weight.requires_grad_(False)

        # self.register_buffer('grid_size_inverse',torch.fill(torch.zeros((1,self.grid_size,self.grid_size,self.num_anchors)),value = (1/self.grid_size)))


    def forward(self, x):

        self.device = x.device
        bs = x.shape[0]
        out = self.conv(x)
        out = self.detect(out)
        out = out.permute(0, 2, 3, 1).flatten(1, 2).view((bs, -1, self.num_anchors, self.num_attributes))

        if self.training:
            pred_obj = torch.nn.Hardsigmoid()(out[..., :1])
            pred_box_txty = torch.nn.Hardsigmoid()(out[..., 1:3])
            pred_box_twth = out[..., 3:5]
            pred_cls = out[..., 5:]
            return torch.cat((pred_obj, pred_box_txty, pred_box_twth, pred_cls), dim=-1)
        else:
            return out
        
            # pred_box = self.transform_pred_box(torch.cat((pred_box_txty, pred_box_twth), dim=-1))

            # # DPU doesn't assign multiplication between two tensor with different channels to CPU. For elemental multiplication to be performed, we need to have the same channel size. 
            # # In order to do that elemental multiplication between two tensor of different channels, we convolve the tensor with smaller channel size with 2D filter of kernel-size 1 and weight 1
            # obj = pred_obj.view(1, self.grid_size * self.grid_size, self.num_anchors, 1).permute(0, 3, 1, 2).reshape(1, 1, self.grid_size* self.grid_size, self.num_anchors)
            # objC = self.replicate(obj)                                # (B, C, S * S, A)
            # objC = objC.view(1, self.num_classes, self.grid_size * self.grid_size, self.num_anchors).permute(0, 2, 3, 1).reshape(1, self.grid_size * self.grid_size, self.num_anchors, self.num_classes)
            # pred_score = torch.mul(objC,torch.nn.Hardsigmoid()(pred_cls))
            # pred_score, pred_label = pred_score.max(dim=-1)
            # pred_out = torch.cat((pred_score.unsqueeze(-1), pred_box, pred_label.unsqueeze(-1)), dim=-1)
            # return pred_out.flatten(1, 2)


    def transform_pred_box(self, pred_box):

        xc = (pred_box[..., 0].view(1,self.grid_size,self.grid_size,3) + self.grid_x.view(1,self.grid_size,self.grid_size,1).expand(1,self.grid_size,self.grid_size,3)) * self.grid_size_inverse
        yc = (pred_box[..., 1].view(1,self.grid_size,self.grid_size,3) + self.grid_y.view(1,self.grid_size,self.grid_size,1).expand(1,self.grid_size,self.grid_size,3)) * self.grid_size_inverse
        
        w = torch.exp(pred_box[..., 2]).view(1,self.grid_size,self.grid_size,3) * self.anchors[:, 0].view(1,1,3).expand(1,self.grid_size,self.grid_size,3)
        h = torch.exp(pred_box[..., 3]).view(1,self.grid_size,self.grid_size,3) * self.anchors[:, 1].view(1,1,3).expand(1,self.grid_size,self.grid_size,3)
        
        return torch.cat((xc.flatten(1,2).unsqueeze(-1), yc.flatten(1,2).unsqueeze(-1), w.flatten(1,2).unsqueeze(-1), h.flatten(1,2).unsqueeze(-1)), dim=-1)


    def set_grid_xy(self, input_size):
        self.grid_size = input_size // self.stride
        grid_x, grid_y = self.set_grid(grid_size=self.grid_size)
        grid_x = grid_x.contiguous().view((1, -1, 1)).to(torch.float32)
        grid_y = grid_y.contiguous().view((1, -1, 1)).to(torch.float32)
        return grid_x,grid_y

    def set_grid(self,grid_size):
        grid_y,grid_x = torch.meshgrid((torch.arange(grid_size),torch.arange(grid_size)),indexing="ij")
        return (grid_x,grid_y)

class YoloHeadDPU(nn.Module):
    def __init__(self, input_size, in_channels, num_classes, anchors):
        super().__init__()
        anchors = torch.tensor(anchors) if not torch.is_tensor(anchors) else anchors
        self.detect_s = DetectLayer_DPU(input_size=input_size, in_channels=in_channels[0]//2, num_classes=num_classes, anchors=anchors[0:3], stride=8)
        self.detect_m = DetectLayer_DPU(input_size=input_size, in_channels=in_channels[1]//2, num_classes=num_classes, anchors=anchors[3:6], stride=16)
        self.detect_l = DetectLayer_DPU(input_size=input_size, in_channels=in_channels[2]//2, num_classes=num_classes, anchors=anchors[6:9], stride=32)

    def forward(self, x):
        C1, C2, C3 = x
        pred_s = self.detect_s(C3)
        pred_m = self.detect_m(C2)
        pred_l = self.detect_l(C1)
        return pred_s, pred_m, pred_l

class YoloHead(nn.Module):
    def __init__(self, input_size, in_channels, num_classes, anchors):
        super().__init__()
        anchors = torch.tensor(anchors) if not torch.is_tensor(anchors) else anchors
        self.detect_s = DetectLayer(input_size=input_size, in_channels=in_channels[0]//2, num_classes=num_classes, anchors=anchors[0:3], stride=8)
        self.detect_m = DetectLayer(input_size=input_size, in_channels=in_channels[1]//2, num_classes=num_classes, anchors=anchors[3:6], stride=16)
        self.detect_l = DetectLayer(input_size=input_size, in_channels=in_channels[2]//2, num_classes=num_classes, anchors=anchors[6:9], stride=32)

    def forward(self, x):
        C1, C2, C3 = x
        pred_s = self.detect_s(C3)
        pred_m = self.detect_m(C2)
        pred_l = self.detect_l(C1)
        return pred_s, pred_m, pred_l

if __name__ == "__main__":
    from backbone import build_backbone
    from neck import FPN

    input_size = 416
    num_classes = 1
    
    anchors = [[0.248,      0.7237237 ],
                [0.36144578, 0.53      ],
                [0.42,       0.9306667 ],
                [0.456,      0.6858006 ],
                [0.488,      0.8168168 ],
                [0.6636637,  0.274     ],
                [0.806,      0.648     ],
                [0.8605263,  0.8736842 ],
                [0.944,      0.5733333 ]]

    backbone, feat_dims = build_backbone()
    neck = FPN(feat_dims=feat_dims)
    head = YoloHead(input_size=input_size, in_channels=feat_dims, num_classes=num_classes, anchors=anchors)

    x = torch.randn(1, 3, input_size, input_size)
    ftrs = backbone(x)
    ftrs = neck(ftrs)
    preds = head(ftrs)
    for pred in preds:
        print(pred.shape)

    head.eval()
    preds = head(ftrs)
    for pred in preds:
        print(pred.shape)

## Import the basic building blocks of YOLOv3
from units import Conv
from backbone import build_backbone
from neck import FPN
from head import YoloHead,YoloHeadDPU


# Import other necessary libraries
import torch
import torch.nn as nn
from pathlib import Path
import sys
import gdown
from decode_predictions import BoxDecoder

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


model_urls = {
    "yolov3-base": 'https://drive.google.com/file/d/1duXcafb2QgORHDO1w-7E1UusLfWInwgA/view',
    "yolov3-spp": None,
    "yolov3-tiny": None,
}


class YOLOv3(nn.Module):
    def __init__(self,input_size,num_classes,anchors,model_type,pretrained = False):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.anchors = torch.tensor(anchors)
        self.model_type = model_type
        self.backbone,self.feat_dims = build_backbone()
        self.neck = FPN(feat_dims=self.feat_dims)
        self.head = YoloHead(in_channels = self.feat_dims,input_size = self.input_size,num_classes = 20,anchors=anchors)

    
        #Todo: Make this naming convention dynamic rather than absolute.
        if pretrained:
            download_path = ROOT / "weights" / f"yolov3-{self.model_type}.pt"
            if not download_path.is_file():
                gdown.download(model_urls[f"yolov3-{self.model_type}"], str(download_path), quiet=False, fuzzy=True)

            ckpt = torch.load(ROOT / "weights" / f"yolov3-{self.model_type}.pt", map_location="cpu")
            self.load_state_dict(ckpt["model_state"], strict=True)
    def forward(self,x):
        ftrs = self.backbone(x)
        ftrs = self.neck(ftrs)
        preds = self.head(ftrs)
        if self.training:
            return preds
        else:
            return torch.cat(preds,dim = 1)
    
    def set_grid_xy(self, input_size):
        self.head.detect_m.set_grid_xy(input_size=input_size)
        self.head.detect_l.set_grid_xy(input_size=input_size)
        self.head.detect_s.set_grid_xy(input_size=input_size)

class YOLOv3_DPU(nn.Module):
    def __init__(self,input_size,num_classes,anchors,model_type,pretrained = False):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.anchors = torch.tensor(anchors)
        self.model_type = model_type
        self.backbone,self.feat_dims = build_backbone()
        self.neck = FPN(feat_dims=self.feat_dims)
        self.head = YoloHeadDPU(in_channels = self.feat_dims,input_size = self.input_size,num_classes = 20,anchors=anchors)

    
        #Todo: Make this naming convention dynamic rather than absolute.
        if pretrained:
            download_path = ROOT / "weights" / f"yolov3-{self.model_type}.pt"
            if not download_path.is_file():
                gdown.download(model_urls[f"yolov3-{self.model_type}"], str(download_path), quiet=False, fuzzy=True)

            ckpt = torch.load(ROOT / "weights" / f"yolov3-{self.model_type}.pt", map_location="cpu")
            self.load_state_dict(ckpt["model_state"], strict=True)
    def forward(self,x):
        ftrs = self.backbone(x)
        ftrs = self.neck(ftrs)
        preds = self.head(ftrs)
        return preds
    
    
    def set_grid_xy(self, input_size):
        self.head.detect_m.set_grid_xy(input_size=input_size)
        self.head.detect_l.set_grid_xy(input_size=input_size)
        self.head.detect_s.set_grid_xy(input_size=input_size)

if __name__ == "__main__":
    input_size = 416
    num_classes = 20
    anchors = [[0.248,      0.7237237 ],
        [0.36144578, 0.53      ],
        [0.42,       0.9306667 ],
        [0.456,      0.6858006 ],
        [0.488,      0.8168168 ],
        [0.6636637,  0.274     ],
        [0.806,      0.648     ],
        [0.8605263,  0.8736842 ],
        [0.944,      0.5733333 ]]
    model_type = "base"

    
    x = torch.randn(1,3,input_size,input_size).to('cpu')
    model = YOLOv3(input_size = input_size,num_classes = num_classes,anchors = anchors,model_type=model_type,pretrained = False)
    model.train()

    output = model(x)
    for preds in output:
        print(preds.shape)
    model.eval()
    output2 = model(x)
    # assert (output2.shape == torch.Size([8,3549,3,6])), "Final Output sized not matched"
    # for preds in output2:
    #     print(preds.shape)

    print(output2.shape)
    
    model_DPU = YOLOv3_DPU(input_size = input_size,num_classes = num_classes,anchors = anchors,model_type=model_type,pretrained = False)
    model_DPU.train()

    output = model_DPU(x)
    for preds in output:
        print(preds.shape)
    
    model_DPU.eval()
    output2_DPU = model_DPU(x)
    # assert (output2.shape == torch.Size([8,3549,3,6])), "Final Output sized not matched"

    for preds in output2_DPU:
        print(preds.shape)
    
    decoded_52 = BoxDecoder(output2_DPU[0],model_DPU.anchors[0:3]).decode_predictions()
    decoded_26 = BoxDecoder(output2_DPU[1],model_DPU.anchors[3:6]).decode_predictions()
    decoded_13 = BoxDecoder(output2_DPU[2],model_DPU.anchors[6:9]).decode_predictions()

    decoded_output = torch.cat((decoded_52,decoded_26,decoded_13),dim = 1)

    print(decoded_output.shape)
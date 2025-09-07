import torch
from pathlib import Path
import sys
import math

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import set_grid


def do_sigmoid(predictions,hard_sig = False):
    if not hard_sig:
        pred_obj = torch.sigmoid(predictions[...,:1])
        pred_box_txty = torch.sigmoid(predictions[...,1:3])
        pred_box_twth = predictions[...,3:5]
        pred_cls = predictions[...,5:]

    else:
        pred_obj = torch.nn.Hardsigmoid(predictions[...,:1])
        pred_box_txty = torch.nn.Hardsigmoid(predictions[...,1:3])
        pred_box_twth = predictions[...,3:5]
        pred_cls = predictions[...,5:]
    
    return torch.cat((pred_obj, pred_box_txty, pred_box_twth, pred_cls), dim=-1)

class BoxDecoder():
    def __init__(self,predictions,anchors):
        self.out = predictions
        self.grid_size = math.sqrt(self.out.shape[1])
        self.anchors = anchors
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def decode_predictions(self):
        pred_obj = torch.nn.Hardsigmoid()(self.out[..., :1])
        pred_box_txty = torch.nn.Hardsigmoid()(self.out[..., 1:3])
        pred_box_twth = self.out[..., 3:5]
        pred_cls = self.out[..., 5:]
        pred_box = self._transform_pred_box(torch.cat((pred_box_txty, pred_box_twth), dim=-1))
        pred_score = pred_obj * torch.sigmoid(pred_cls)
        pred_score, pred_label = pred_score.max(dim=-1)
        pred_out = torch.cat((pred_score.unsqueeze(-1), pred_box, pred_label.unsqueeze(-1)), dim=-1)
        return pred_out.flatten(1,2)

    def _transform_pred_box(self, pred_box):
            self.grid_x,self.grid_y = self._set_grid_xy(grid_size=self.grid_size)
            xc = (pred_box[..., 0] + self.grid_x.to(self.device)) / self.grid_size
            yc = (pred_box[..., 1] + self.grid_y.to(self.device)) / self.grid_size
            w = torch.exp(pred_box[..., 2]) * self.anchors[:, 0].to(self.device)
            h = torch.exp(pred_box[..., 3]) * self.anchors[:, 1].to(self.device)
            return torch.cat((xc.unsqueeze(-1), yc.unsqueeze(-1), w.unsqueeze(-1), h.unsqueeze(-1)), dim=-1)
    
    def _set_grid_xy(self, grid_size):
        grid_y,grid_x = torch.meshgrid((torch.arange(grid_size),torch.arange(grid_size)),indexing="ij")
        grid_x = grid_x.contiguous().view((1, -1, 1)).to(torch.float32)
        grid_y = grid_y.contiguous().view((1, -1, 1)).to(torch.float32)
        return grid_x,grid_y


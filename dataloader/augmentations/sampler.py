import numpy as np
from typing import Optional,Sequence,Tuple



MEAN = 0.485, 0.456, 0.406 # RGB
STD = 0.229, 0.224, 0.225 # RGB

def transform_xcycwh_to_x1y1x2y2(boxes, clip_max=None):
    x1y1 = boxes[:, :2] - boxes[:, 2:] / 2
    x2y2 = boxes[:, :2] + boxes[:, 2:] / 2
    x1y1x2y2 = np.concatenate((x1y1, x2y2), axis=1)
    return x1y1x2y2.clip(min=0, max=clip_max if clip_max is not None else 1)


def scale_to_original(boxes,scale_w,scale_h):
    boxes[:,[0,2]] *= scale_w
    boxes[:,[1,3]] *= scale_h
    return boxes.round(2)

def clip_xyxy(boxes:np.ndarray,W:int,H:int) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    
    boxes[:,[0,2]] = np.clip(boxes[:,[0,2]],0,max(0,W - 1))
    boxes[:,[1,3]]= np.clip(boxes[:,[1,3]],0,max(0,H - 1))
    return boxes

class SampleProvider:
    """
    Minimal Provider that fetches a random raw sample from Dataset.
    Uses get_GT_item function from Dataset class.
    
    """
    #Todo: Enable option to fetch the boundary box in any format desired by the user.
    def __init__(self,dataset,rng:Optional[np.random.RandomState] = None):
        self.ds = dataset
        self.rng = rng or np.random.RandomState()

    def sample(self) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        """Returns (image_rgb_uint8,boxes_xyxy_abs,labels)"""
        idx = self.rng.randint(0,len(self.ds))

        _,img,lbl = self.ds.get_GT_item(idx)
        H,W = img.shape[:2]
        if len(lbl):
            classes = lbl[:,0]
            boxes_cxcywh = lbl[:,1:]
            boxes_x1y2x2y2 = transform_xcycwh_to_x1y1x2y2(boxes=boxes_cxcywh,clip_max=1)
            boxes_x1y2x2y2_abs = scale_to_original(boxes = boxes_x1y2x2y2,scale_w = W,scale_h = H)
            boxes = clip_xyxy(boxes_x1y2x2y2_abs,W = W,H = H)
        else:
            boxes = np.zeros((0,4),np.float32)
            classes = np.zeros((0,4),np.float32)
        
        return img,boxes,classes
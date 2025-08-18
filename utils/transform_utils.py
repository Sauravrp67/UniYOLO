import numpy as np
import cv2
import torch
from typing import Optional,Sequence,Tuple
from general import transform_xcycwh_to_x1y1x2y2,scale_to_original

MEAN = 0.485, 0.456, 0.406 # RGB
STD = 0.229, 0.224, 0.225 # RGB


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

def clip_xyxy(boxes:np.ndarray,W:int,H:int) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    
    boxes[:,[0,2]] = np.clip(boxes[:,[0,2]],0,max(0,W - 1))
    boxes[:,[1,3]]= np.clip(boxes[:,[1,3]],0,max(0,H - 1))
    return boxes

def to_tensor(image):
    image = np.ascontiguousarray(image.transpose(2, 0, 1))
    return torch.from_numpy(image).float()


def to_image(tensor, mean=MEAN, std=STD):
    denorm_tensor = tensor.clone()
    for t, m, s in zip(denorm_tensor, mean, std):
        t.mul_(s).add_(m)
    denorm_tensor.clamp_(min=0, max=1.)
    denorm_tensor *= 255
    image = denorm_tensor.permute(1,2,0).numpy().astype(np.uint8)
    return image


def denormalize(image, mean=MEAN, std=STD):
    image_temp = image.copy()
    image_temp *= std
    image_temp += mean
    image_temp *= 255.
    return image_temp.astype(np.uint8)
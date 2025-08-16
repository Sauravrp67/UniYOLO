import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


import numpy as np
import cv2
from typing import Optional,Tuple
from utils import scale_to_original,transform_xcycwh_to_x1y1x2y2,SampleProvider



def _clip_xyxy(boxes: np.ndarray, W: int, H: int) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    
    boxes[:,[0,2]] = np.clip(boxes[:,[0,2]],0,max(0,W - 1))
    boxes[:,[1,3]] = np.clip(boxes[:,[1,3]],0,max(0,H - 1))
    return boxes

# class CutMixProvider:
#     """
#     This class fetches a random *raw* sample from your dataset(no augmentation). It uses the Dataset's get_image/get_label helpers
#     to randomly fetch image and label.

#     Usage:
#         provider = CutMixProvider(Dataset)
#         cm = RandomCutMix(provider=provider,p =0.2)
#     """
#     def __init__(self,dataset,rng:Optional[np.random.RandomState] = None):
#         self.ds = dataset
#         self.rng = rng or np.random.RandomState()

#     def sample(self) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
#         """Returns (image_rgb_uint8,boxes_xyxy_abs,labels)"""
#         idx = self.rng.randint(0,len(self.ds))

#         _,img,lbl = self.ds.get_GT_item(idx)
#         H,W = img.shape[:2]
#         classes = lbl[:,0]
#         boxes_cxcywh = lbl[:,1:]
#         boxes_x1y1x2y2 = transform_xcycwh_to_x1y1x2y2(boxes_cxcywh,clip_max=1)
#         boxes_x1y1x2y2_abs = scale_to_original(boxes_x1y1x2y2,scale_w =W,scale_h = H)
#         boxes = _clip_xyxy(boxes_x1y1x2y2_abs,W = W,H = H)
#         return img,boxes,classes
    
class RandomCutMix:
    """
    Two-image detection Cutmix:
     -selects a random rectangle in the current image
     -paste a same-sized crop from a random other image
     - performs BBox transformation.

     Assumes the __call__ receives:
        - image: HxWxC (uint8 or float32, RGB)
        - boxes: (N,4) numpy array in absolute XminYmixXmaxYmax format.

    Must be performed after Photometric transformation and before Affine and Geometric transformations.
    """
    def __init__(
            self,
            provider:SampleProvider,
            p:float = 0.2,
            min_ratio:float = 0.3,
            max_ratio:float = 0.7,
            drop_occluded:bool = True,
            occ_thresh:float = 0.6,
            min_keep_size:int = 2
    ):
        self.provider = provider
        self.p = float(p)
        self.min_ratio = float(min_ratio)
        self.max_ratio = float(max_ratio)
        self.drop_occluded = bool(drop_occluded)
        self.occ_thresh = float(occ_thresh)
        self.min_keep_size = int(min_keep_size)

    def __call__(self, image: np.ndarray, boxes: np.ndarray, labels: Optional[np.ndarray] = None):
        # no-op cases
        if self.provider is None or np.random.rand() > self.p:
            return image, boxes, labels

        H, W = image.shape[:2]
        if H < 2 or W < 2:
            return image, boxes, labels

        # Choose paste rect on img1
        rw = int(np.random.uniform(self.min_ratio, self.max_ratio) * W)
        rh = int(np.random.uniform(self.min_ratio, self.max_ratio) * H)
        rw = max(self.min_keep_size, min(rw, W))
        rh = max(self.min_keep_size, min(rh, H))

        rx = np.random.randint(0, max(1, W - rw + 1))
        ry = np.random.randint(0, max(1, H - rh + 1))
        paste_rect = np.array([rx, ry, rx + rw, ry + rh], dtype=np.int32)

        # Fetch second image
        img2, b2, l2 = self.provider.sample()  # img2 uint8 RGB
        H2, W2 = img2.shape[:2]
        # Ensure img2 is big enough to supply a rw x rh crop: resize if needed
        need_s = max(rw / max(1, W2), rh / max(1, H2))
        if need_s > 1.0:
            newW2 = int(W2 * need_s + 1e-6)
            newH2 = int(H2 * need_s + 1e-6)
            img2 = cv2.resize(img2, (newW2, newH2), interpolation=cv2.INTER_LINEAR)
            b2 = b2 * need_s
            W2, H2 = newW2, newH2

        # Pick source crop from img2 of size (rw,rh)
        sx = np.random.randint(0, max(1, W2 - rw + 1))
        sy = np.random.randint(0, max(1, H2 - rh + 1))
        src_rect = np.array([sx, sy, sx + rw, sy + rh], dtype=np.int32)

        # Paste img2 crop onto image1
        patch = img2[sy:sy + rh, sx:sx + rw]
        
        if image.dtype != patch.dtype:
            patch = patch.astype(image.dtype,copy = False)
        patch_image = image
        patch_image[ry:ry + rh,rx :rx + rw] = patch

        if len(b2):
            ix1 = np.maximum(b2[:,0],src_rect[0])
            iy1 = np.maximum(b2[:,1],src_rect[1])
            ix2 = np.minimum(b2[:,2],src_rect[2])
            iy2 = np.minimum(b2[:,3],src_rect[3])
            
            # Here, if the two boxes don't intersect then, x2-y2 becomes negative and we clamp at 0,
            # so iw, and ih would be 0
            iw = np.clip(ix2 - ix1,0,None)
            ih = np.clip(iy2 - iy1,0,None)
            keep = (iw >= self.min_keep_size) & (ih >= self.min_keep_size)

            if keep.any():
                inter = np.stack([ix1[keep], iy1[keep], ix2[keep], iy2[keep]], axis=1)
                # translate inter to paste location on image1
                inter[:, [0, 2]] = inter[:, [0, 2]] - sx + rx
                inter[:, [1, 3]] = inter[:, [1, 3]] - sy + ry
                inter = _clip_xyxy(inter, W, H)
                new_boxes2 = inter
                new_labels2 = l2[keep].astype(labels.dtype if labels is not None else np.float32)
            else:
                new_boxes2 = np.zeros((0, 4), np.float32)
                new_labels2 = np.zeros((0,), np.float32)
        else:
            new_boxes2 = np.zeros((0, 4), np.float32)
            new_labels2 = np.zeros((0,), np.float32)

        if len(boxes) and self.drop_occluded:
            b1 = boxes.astype(np.float32)
            bx1 = np.maximum(b1[:, 0], paste_rect[0])
            by1 = np.maximum(b1[:, 1], paste_rect[1])
            bx2 = np.minimum(b1[:, 2], paste_rect[2])
            by2 = np.minimum(b1[:, 3], paste_rect[3])
            bw = np.clip(bx2 - bx1, 0, None)
            bh = np.clip(by2 - by1, 0, None)
            inter = bw * bh
            area = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
            with np.errstate(divide='ignore', invalid='ignore'):
                occ = np.where(area > 0, inter / area, 0.0)
            keep1 = occ <= self.occ_thresh
            boxes1 = boxes[keep1]
            labels1 = labels[keep1] if labels is not None else None
        else:
            boxes1, labels1 = boxes, labels

        # Merge
        if len(new_boxes2):
            out_boxes = np.concatenate([boxes1, new_boxes2], axis=0) if len(boxes1) else new_boxes2
            if labels1 is not None:
                out_labels = np.concatenate([labels1, new_labels2], axis=0) if len(labels1) else new_labels2
            else:
                out_labels = None
        else:
            out_boxes, out_labels = boxes1, labels1

        return image,out_boxes,out_labels
        

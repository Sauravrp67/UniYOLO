# dataloader/augmentations/mixup.py
import numpy as np
import cv2
from typing import Optional, Tuple
from sampler import SampleProvider

# class MixUpProvider:
#     """
#     Minimal provider for MixUp that fetches a random raw sample
#     (RGB image, XYXY abs boxes, labels) from your dataset helpers.
#     """
#     def __init__(self, dataset, rng: Optional[np.random.RandomState] = None):
#         self.ds = dataset
#         self.rng = rng or np.random.RandomState()

#     def sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#         j = self.rng.randint(0, len(self.ds))
#         _, img = self.ds.get_image(j)      # RGB uint8
#         H, W = img.shape[:2]
#         lbl = self.ds.get_label(j)         # YOLO [cls,cx,cy,w,h] normalized
#         if len(lbl):
#             cls = lbl[:, 0].astype(np.float32)
#             cx  = lbl[:, 1] * W; cy = lbl[:, 2] * H
#             ww  = lbl[:, 3] * W; hh = lbl[:, 4] * H
#             x1 = cx - ww * 0.5; y1 = cy - hh * 0.5
#             x2 = cx + ww * 0.5; y2 = cy + hh * 0.5
#             boxes = np.stack([x1, y1, x2, y2], 1).astype(np.float32)
#             labels = cls
#         else:
#             boxes  = np.zeros((0, 4), np.float32)
#             labels = np.zeros((0,),  np.float32)
#         return img, boxes, labels


class RandomMixUp:
    """
    MixUp for detection (two images blended linearly).
    - Inputs/outputs: image HxWxC (uint8/float32), boxes (N,4) XYXY abs, labels (N,)
    - λ ~ Beta(alpha, alpha). Labels are simply concatenated (no per-box weights).
    - Place after ToXminYminXmaxYmax + ToAbsoluteCoords.
      Common YOLO order: [Mosaic] -> (MixUp) -> RandomPerspective -> ...
    """
    def __init__(self,
                 provider: SampleProvider,
                 p: float = 0.15,
                 alpha: float = 0.20,
                 pad_value: Tuple[float, float, float] = 114,
                 apply_lambda_to_labels = False,
                 min_visible_lambda:float = 0.20):
        self.provider = provider
        self.p = float(p)
        self.alpha = float(alpha)
        self.pad_value = tuple((np.array(pad_value) * 255).astype(np.uint8).tolist())
        self.apply_lambda_to_labels = bool(apply_lambda_to_labels)
        self.min_visible_lambda = float(min_visible_lambda)
    def __call__(self, image: np.ndarray, boxes: np.ndarray, labels: Optional[np.ndarray] = None):
        if self.provider is None or np.random.rand() > self.p:
            return image, boxes, labels

        img1, b1 = image, boxes
        l1 = labels if labels is not None else np.zeros((0,), np.float32)

        # sample partner
        img2, b2, l2 = self.provider.sample()

        # make same canvas size (pad both to max H/W)
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])

        can1 = self._pad_to(img1, h, w, self.pad_value)
        can2 = self._pad_to(img2, h, w, self.pad_value)

        # adjust boxes for padded canvas
        b1_adj = b1.copy()
        b2_adj = b2.copy()
        # (no translation needed since we anchor both at (0,0))

        # mix
        lam = np.random.beta(self.alpha, self.alpha)

        # keep dtype
        if can1.dtype != np.float32:
            can1 = can1.astype(np.float32)
        if can2.dtype != np.float32:
            can2 = can2.astype(np.float32)
        mixed = (lam * can1 + (1.0 - lam) * can2)

        # cast back if original was uint8
        out = mixed.astype(image.dtype, copy=False) if image.dtype != np.float32 else mixed

        # print(f"Lambda:{lam},1 - lambda:{1 - lam}")
        #---weak-side filtering to avoid "ghost" boxes----
        keep1 = True
        keep2 = True

        if lam < (1.0 - lam): # img2 dominates
            keep1 = lam >= self.min_visible_lambda
        else: #Image1 dominates
            keep2 = (1.0 - lam) >= self.min_visible_lambda

        # print(f"Keep1:{keep1},keep2:{keep2}")
        
        b1_kept = b1_adj if keep1 else np.zeros((0,4), np.float32)
        l1_kept = l1 if keep1 else np.zeros((0,),  np.float32)
        b2_kept = b2_adj if keep2 else np.zeros((0,4), np.float32)
        l2_kept = l2 if keep2 else np.zeros((0,),  np.float32) 

        # labels: concat (most YOLO losses don’t need λ per box)
        # labels (optionally weighted)
        if self.apply_lambda_to_labels:
            # store as 2-column: [cls_id, weight]
            if len(l1_kept):
                l1w = np.full((len(l1_kept),), lam, dtype=np.float32)
                lab1 = np.stack([l1_kept, l1w], 1)
            else:
                lab1 = np.zeros((0,2), np.float32)

            if len(l2_kept):
                l2w = np.full((len(l2_kept),), 1.0 - lam, dtype=np.float32)
                lab2 = np.stack([l2_kept, l2w], 1)
            else:
                lab2 = np.zeros((0,2), np.float32)

            out_labels = np.concatenate([lab1, lab2], 0)
        else:
            out_labels = np.concatenate([l1_kept, l2_kept], 0)

        out_boxes = np.concatenate([b1_kept, b2_kept], 0)

        return out, out_boxes, out_labels

    @staticmethod
    def _pad_to(image: np.ndarray, H: int, W: int, pad_value=114) -> np.ndarray:
        h, w = image.shape[:2]
        if h == H and w == W:
            return image
        canvas = np.full((H, W, image.shape[2]), pad_value, dtype=image.dtype) if image.ndim == 3 \
                 else np.full((H, W), pad_value, dtype=image.dtype)
        canvas[:h, :w] = image
        return canvas

import numpy as np
import cv2
from typing import Optional,Tuple,Sequence
from utils import scale_to_original,transform_xcycwh_to_x1y1x2y2,clip_xyxy,SampleProvider



class RandomMosaic:
    """
    4-image Mosaic (YOLOv5-style) with occlusion filtering.

    Returns:
        canvas (2S x 2S), boxes_xyxy_abs, labels

    Typical pipeline placement:
        Mosaic -> RandomPerspective/Affine -> HFlip -> Cutout -> Letterbox(S) -> Normalize
        (Usually skip Expand/IoU-Crop when Mosaic is active.)

    Args:
        provider: MosaicProvider (required)
        output_size: base size S (int or (S,S)). Canvas will be (2S, 2S)
        scale_range: per-tile random scale factor (lo, hi)
        center_ratio_range: choose mosaic center in [lo*S, hi*S]
        pad_value: background color (114 or mean tuple)
        min_box_wh: drop tiny boxes (pixels)
        min_visible: min visible fraction of a box within the pasted crop to keep it (0..1)
        enforce_min_fill: if >0, floor the per-tile scale so the tile short side >= enforce_min_fill*S
        p: probability to apply (set to 1.0 if you gate outside with a selector)
    """
    def __init__(self,
                 provider: SampleProvider,
                 output_size: int,
                 scale_range: Tuple[float, float] = (0.5, 1.5),
                 center_ratio_range: Tuple[float, float] = (0.5, 1.5),
                 pad_value: Tuple[float, float, float] = 114,
                 min_box_wh: int = 2,
                 min_visible: float = 0.5,
                 enforce_min_fill: float = 0.0,
                 p: float = 1.0):
        self.provider = provider
        self.S = int(output_size[0] if isinstance(output_size, (list, tuple)) else output_size)
        self.scale_range = tuple(float(x) for x in scale_range)
        self.center_ratio_range = tuple(float(x) for x in center_ratio_range)
        self.pad_value = tuple((np.array(pad_value) * 255).astype(np.uint8).tolist())
        self.min_box_wh = int(min_box_wh)
        self.min_visible = float(min_visible)
        self.enforce_min_fill = float(enforce_min_fill)
        self.p = float(p)

    def __call__(self, image: np.ndarray, boxes: np.ndarray, labels: Optional[np.ndarray] = None):
        if self.provider is None or np.random.rand() > self.p:
            return image, boxes, labels

        S = self.S
        Hc, Wc = 2 * S, 2 * S  # canvas size

        # mosaic center
        yc = int(np.random.uniform(self.center_ratio_range[0] * S,
                                   self.center_ratio_range[1] * S))
        xc = int(np.random.uniform(self.center_ratio_range[0] * S,
                                   self.center_ratio_range[1] * S))

        # tiles: current + 3 random
        imgs = [image]
        bxs  = [boxes if boxes is not None else np.zeros((0, 4), np.float32)]
        lbs  = [labels if labels is not None else np.zeros((0,), np.float32)]
        for _ in range(3):
            img2, bx2, lb2 = self.provider.sample()
            imgs.append(img2); bxs.append(bx2); lbs.append(lb2)

        # canvas
        if isinstance(self.pad_value, (tuple, list, np.ndarray)):
            canvas = np.zeros((Hc, Wc, 3), dtype=imgs[0].dtype)
            canvas[...] = np.array(self.pad_value, dtype=imgs[0].dtype)
        else:
            C = imgs[0].shape[2] if imgs[0].ndim == 3 else 1
            canvas = np.full((Hc, Wc, C), self.pad_value, dtype=imgs[0].dtype)

        all_boxes = []
        all_labels = []

        for k in range(4):
            img = imgs[k]
            boxes_k = bxs[k]
            labels_k = lbs[k]

            h, w = img.shape[:2]
            lo, hi = self.scale_range
            
            if self.enforce_min_fill > 0.0:
                short = max(1, min(h, w))
                r_min_req = (self.enforce_min_fill * S) / short
                lo = max(lo, r_min_req)
                hi = max(hi, lo + 1e-6)
            r = np.random.uniform(lo, hi)

            if r != 1.0:
                nw, nh = int(w * r), int(h * r)
                img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
                if len(boxes_k):
                    boxes_k = boxes_k.astype(np.float32).copy() * r
            else:
                nw, nh = w, h

            if k == 0:  # top-left
                x1a = max(xc - nw, 0); y1a = max(yc - nh, 0)
                x2a = xc;              y2a = yc
                x1b = nw - (x2a - x1a); y1b = nh - (y2a - y1a)
                x2b = nw;               y2b = nh
            elif k == 1:  # top-right
                x1a = xc;               y1a = max(yc - nh, 0)
                x2a = min(xc + nw, Wc); y2a = yc
                x1b = 0;                y1b = nh - (y2a - y1a)
                x2b = min(nw, x2a - x1a); y2b = nh
            elif k == 2:  # bottom-left
                x1a = max(xc - nw, 0); y1a = yc
                x2a = xc;               y2a = min(yc + nh, Hc)
                x1b = nw - (x2a - x1a); y1b = 0
                x2b = nw;               y2b = min(nh, y2a - y1a)
            else:         # bottom-right
                x1a = xc;               y1a = yc
                x2a = min(xc + nw, Wc); y2a = min(yc + nh, Hc)
                x1b = 0;                y1b = 0
                x2b = min(nw, x2a - x1a); y2b = min(nh, y2a - y1a)

            pw, ph = (x2a - x1a), (y2a - y1a)
            cw, ch = (x2b - x1b), (y2b - y1b)
            if pw <= 0 or ph <= 0 or cw <= 0 or ch <= 0:
                continue

            # paste
            canvas[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

            # --- boxes: intersect -> occlusion filter -> shift ---
            if len(boxes_k):
                crop = np.array([x1b, y1b, x2b, y2b], dtype=np.float32)  # in TILE coords

                b = boxes_k.astype(np.float32).copy()
                ix1 = np.maximum(b[:, 0], crop[0])
                iy1 = np.maximum(b[:, 1], crop[1])
                ix2 = np.minimum(b[:, 2], crop[2])
                iy2 = np.minimum(b[:, 3], crop[3])

                iw = np.clip(ix2 - ix1, 0, None)
                ih = np.clip(iy2 - iy1, 0, None)
                inter = iw * ih

                area = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
                with np.errstate(divide='ignore', invalid='ignore'):
                    vis = np.where(area > 0, inter / area, 0.0)

                keep = (iw >= self.min_box_wh) & (ih >= self.min_box_wh) & (vis >= self.min_visible)
                if keep.any():
                    inter_boxes = np.stack([ix1[keep], iy1[keep], ix2[keep], iy2[keep]], 1)
                    # shift from tile-crop to canvas
                    inter_boxes[:, [0, 2]] += (x1a - x1b)
                    inter_boxes[:, [1, 3]] += (y1a - y1b)

                    all_boxes.append(inter_boxes)
                    all_labels.append(labels_k[keep])

        # --- merge & final clean ---
        if len(all_boxes):
            boxes_out = np.concatenate(all_boxes, 0).astype(np.float32)
            labels_out = np.concatenate(all_labels, 0).astype(np.float32)
            boxes_out = clip_xyxy(boxes_out, Wc, Hc)
            # tiny filter again (after clip)
            wh = boxes_out[:, 2:4] - boxes_out[:, 0:2]
            keep = (wh[:, 0] >= self.min_box_wh) & (wh[:, 1] >= self.min_box_wh)
            boxes_out = boxes_out[keep]
            labels_out = labels_out[keep] if len(labels_out) else labels_out
        else:
            boxes_out = np.zeros((0, 4), np.float32)
            labels_out = np.zeros((0,), np.float32)

        return canvas, boxes_out, labels_out

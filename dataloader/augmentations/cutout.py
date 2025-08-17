import numpy as np
import cv2

class CustomCutout:
    """
    NumPy-only multi-hole Cutout with cumulative occlusion and bbox removal.
    - boxes: (N,4) absolute XYXY
    - fill_value: scalar or (R,G,B) in the SAME scale/dtype as `image` (e.g., 0–255 uint8)
    """

    def __init__(self,
                 fill_value=0,
                 bbox_removal_threshold=0.70,
                 min_cutout_pct=5,     # percent of min(H,W)
                 max_cutout_pct=20,
                 min_holes=2,
                 max_holes=5,
                 p=0.5,
                 verbose=True):
        self.fill_value = fill_value
        self.bbox_removal_threshold = float(bbox_removal_threshold)
        self.min_cutout_pct = int(min_cutout_pct)
        self.max_cutout_pct = int(max_cutout_pct)
        self.min_holes = int(min_holes)
        self.max_holes = int(max_holes)
        self.p = float(p)
        self.verbose = verbose

    def __call__(self, image, boxes, labels=None, allowed_rect=None):
        # no-op cases
        if image is None or boxes is None or len(boxes) == 0:
            return image, boxes, labels
        if np.random.rand() > self.p:
            return image, boxes, labels

        img = image.copy()
        H, W = img.shape[:2]
        C = 1 if img.ndim == 2 else img.shape[2]

        # hole size range in pixels, relative to current image
        min_side = max(1, min(H, W))
        min_cut = max(1, int(self.min_cutout_pct * 0.01 * min_side))
        max_cut = max(min_cut, int(self.max_cutout_pct * 0.01 * min_side))

        # number of holes
        num_holes = np.random.randint(self.min_holes, self.max_holes + 1)

        # cumulative occlusion per box
        b = boxes.astype(np.float32)
        bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        bw = np.clip(bx2 - bx1, 0, None)
        bh = np.clip(by2 - by1, 0, None)
        box_area = bw * bh
        occ_accum = np.zeros(len(b), dtype=np.float32)

        # helper: place a hole fully inside either `allowed_rect` (content) or the whole image
        if allowed_rect is not None:
            l, t, r, btm = map(int, allowed_rect)
            l, t = max(0, l), max(0, t)
            r, btm = min(W, r), min(H, btm)
        else:
            l, t, r, btm = 0, 0, W, H

        for _ in range(num_holes):
            size = np.random.randint(min_cut, max_cut + 1)

            max_x = max(l, r - size)
            max_y = max(t, btm - size)
            if max_x < l or max_y < t:
                # hole would not fit; skip this hole
                continue

            x0 = np.random.randint(l, max_x + 1)
            y0 = np.random.randint(t, max_y + 1)
            x1 = x0 + size
            y1 = y0 + size

            # apply fill
            if C == 1:
                img[y0:y1, x0:x1] = self.fill_value
            else:
                if np.isscalar(self.fill_value):
                    img[y0:y1, x0:x1, :] = self.fill_value
                else:
                    fv = np.array(self.fill_value, dtype=img.dtype).reshape(1, 1, -1)
                    img[y0:y1, x0:x1, :] = fv

            # accumulate occlusion for each box (intersection with this hole)
            ix1 = np.maximum(bx1, x0)
            iy1 = np.maximum(by1, y0)
            ix2 = np.minimum(bx2, x1)
            iy2 = np.minimum(by2, y1)
            iw = np.clip(ix2 - ix1, 0, None)
            ih = np.clip(iy2 - iy1, 0, None)
            inter = iw * ih
            occ_accum += inter

        # final keep/drop based on cumulative occlusion
        with np.errstate(divide='ignore', invalid='ignore'):
            occ_ratio = np.where(box_area > 0, occ_accum / box_area, 0.0)

        keep_mask = occ_ratio <= self.bbox_removal_threshold
        boxes_kept = boxes[keep_mask]
        labels_kept = labels[keep_mask] if labels is not None else None

        if self.verbose:
            print(f"[Cutout] holes={num_holes}, size_px=[{min_cut},{max_cut}], "
                  f"kept={keep_mask.sum()}/{len(keep_mask)}")

        return img, boxes_kept, labels_kept
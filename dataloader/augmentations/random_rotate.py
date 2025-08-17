import numpy as np
import cv2

class RandomRotate:
    def __init__(self,fill_value=(123,117,104), max_deg=10, p=0.5,
                 keep_mode="corners", min_box_area=4, pad_canvas=True, max_tries=20):
        self.max_deg = float(max_deg)
        self.p = float(p)
        self.fill_value = tuple((np.array(fill_value) * 255).astype(np.uint8).tolist())
        self.keep_mode = keep_mode     
        self.min_box_area = min_box_area
        self.pad_canvas = pad_canvas    
        self.max_tries = max_tries

    def __call__(self, image, boxes, labels=None):
        if image is None or boxes is None or len(boxes)==0 or np.random.rand()>self.p:
            return image, boxes, labels

        H, W = image.shape[:2]
        for _ in range(self.max_tries):
            angle = np.random.uniform(-self.max_deg, self.max_deg)
            M = cv2.getRotationMatrix2D((W/2.0, H/2.0), angle, 1.0)
            cos, sin = abs(M[0,0]), abs(M[0,1])

            if self.pad_canvas:
                # compute new canvas size that fully contains the rotated image
                newW = int(W * cos + H * sin)
                newH = int(W * sin + H * cos)
                # adjust translation so the original center maps to new center
                M[0,2] += (newW/2.0) - W/2.0
                M[1,2] += (newH/2.0) - H/2.0
                outW, outH = newW, newH
            else:
                outW, outH = W, H

            rotated = cv2.warpAffine(
                image, M, (outW, outH),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=self.fill_value
            )

            # transform boxes (rotate 4 corners, then AABB)
            b = boxes.astype(np.float32)
            corners = np.stack([
                np.stack([b[:,0], b[:,1]], axis=1),
                np.stack([b[:,2], b[:,1]], axis=1),
                np.stack([b[:,2], b[:,3]], axis=1),
                np.stack([b[:,0], b[:,3]], axis=1),
            ], axis=1)                              # (N,4,2)
            hom = np.concatenate([corners, np.ones((*corners.shape[:2],1), np.float32)], axis=2)  # (N,4,3)
            rot = hom @ M.T                          # (N,4,2)

            x_min = np.min(rot[:,:,0], axis=1)
            y_min = np.min(rot[:,:,1], axis=1)
            x_max = np.max(rot[:,:,0], axis=1)
            y_max = np.max(rot[:,:,1], axis=1)

            # keep-inside checks
            keep = np.ones(len(b), dtype=bool)
            if self.keep_mode == "corners":
                # require all 4 corners inside canvas
                inside = (rot[:,:,0] >= 0) & (rot[:,:,0] < outW) & (rot[:,:,1] >= 0) & (rot[:,:,1] < outH)
                keep &= inside.all(axis=1)
            elif self.keep_mode == "center":
                cx = (b[:,0] + b[:,2]) * 0.5
                cy = (b[:,1] + b[:,3]) * 0.5
                centers = np.stack([cx, cy, np.ones_like(cx)], axis=1) @ M.T
                keep &= (centers[:,0] >= 0) & (centers[:,0] < outW) & (centers[:,1] >= 0) & (centers[:,1] < outH)

            # clip and size filter
            x_min = np.clip(x_min, 0, outW-1); y_min = np.clip(y_min, 0, outH-1)
            x_max = np.clip(x_max, 0, outW-1); y_max = np.clip(y_max, 0, outH-1)
            new_boxes = np.stack([x_min, y_min, x_max, y_max], axis=1)

            w = new_boxes[:,2] - new_boxes[:,0]; h = new_boxes[:,3] - new_boxes[:,1]
            keep &= (w*h) >= self.min_box_area

            if not keep.any():
                # resample angle if nothing valid and we care about keeping objects
                continue
            print(f"Angle:{angle}")
            return rotated, new_boxes[keep], (labels[keep] if labels is not None else None)
        
        # fallback: give up rotation if we failed too many times
        return image, boxes, labels
import numpy as np
import cv2

class RandomShear:
    """
    Shear image + XYXY boxes via affine transform.
    - boxes must be absolute pixel coords (N,4) [x1,y1,x2,y2]
    - canvas_mode: "crop" (keep HxW) or "expand" (fit full sheared image)
    - keep_mode: "center" (keep if center inside) or "corners" (all 4 inside)
    """
    def __init__(self,
                 shear_x=(-0.2, 0.2),      # tan(theta_x); +/- ~11.3 degrees by default
                 shear_y=(0.0, 0.0),       # set e.g. (-0.1, 0.1) to enable vertical shear
                 p=0.5,
                 fill_value=(123,117,104), # match your uint8 image scale
                 canvas_mode="crop",       # "crop" | "expand"
                 keep_mode="center",       # "center" | "corners"
                 min_box_area=4,
                 max_tries=20):
        self.shear_x = shear_x
        self.shear_y = shear_y
        self.p = float(p)
        self.fill_value = tuple((np.array(fill_value) * 255).astype(np.uint8).tolist())
        self.canvas_mode = canvas_mode
        self.keep_mode = keep_mode
        self.min_box_area = float(min_box_area)
        self.max_tries = int(max_tries)

    @staticmethod
    def _build_shear_matrix(W, H, shx, shy):
        # Build 3x3 affine that shears around the image center
        cx, cy = W/2.0, H/2.0
        T1 = np.array([[1,0,-cx],
                       [0,1,-cy],
                       [0,0,  1]], dtype=np.float32)
        S  = np.array([[1,shx,0],
                       [shy,1,0],
                       [0,  0,1]], dtype=np.float32)
        T2 = np.array([[1,0,cx],
                       [0,1,cy],
                       [0,0, 1]], dtype=np.float32)
        M3 = T2 @ S @ T1   # 3x3
        return M3

    def __call__(self, image, boxes, labels=None):
        if image is None or boxes is None or len(boxes)==0 or np.random.rand() > self.p:
            return image, boxes, labels

        H, W = image.shape[:2]
        b = boxes.astype(np.float32)

        for _ in range(self.max_tries):
            shx = np.random.uniform(*self.shear_x)
            shy = np.random.uniform(*self.shear_y)
            M3  = self._build_shear_matrix(W, H, shx, shy)

            if self.canvas_mode == "expand":
                # Transform image corners to get required output size
                corners_img = np.array([[0,0,1],
                                        [W,0,1],
                                        [W,H,1],
                                        [0,H,1]], dtype=np.float32)     # (4,3)
                rc = (corners_img @ M3.T)[:, :2]                          # (4,2)
                minx, miny = rc[:,0].min(), rc[:,1].min()
                maxx, maxy = rc[:,0].max(), rc[:,1].max()
                outW = int(np.ceil(maxx - minx))
                outH = int(np.ceil(maxy - miny))
                # shift so top-left is (0,0)
                M3[0,2] -= minx
                M3[1,2] -= miny
            else:
                outW, outH = W, H

            M = M3[:2, :]  # 2x3 for warpAffine

            # warp image
            sheared = cv2.warpAffine(
                image, M, (outW, outH),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=self.fill_value
            )

            # transform 4 corners of each box
            corners = np.stack([
                np.stack([b[:,0], b[:,1]], axis=1),
                np.stack([b[:,2], b[:,1]], axis=1),
                np.stack([b[:,2], b[:,3]], axis=1),
                np.stack([b[:,0], b[:,3]], axis=1),
            ], axis=1)  # (N,4,2)
            hom = np.concatenate([corners, np.ones((*corners.shape[:2],1), np.float32)], axis=2)  # (N,4,3)
            rot = hom @ M3.T  # (N,4,2)

            # keep checks
            keep = np.ones(len(b), dtype=bool)
            if self.keep_mode == "corners":
                inside = (rot[...,0] >= 0) & (rot[...,0] < outW) & (rot[...,1] >= 0) & (rot[...,1] < outH)
                keep &= inside.all(axis=1)
            elif self.keep_mode == "center":
                cx = (b[:,0] + b[:,2]) * 0.5
                cy = (b[:,1] + b[:,3]) * 0.5
                centers = np.stack([cx, cy, np.ones_like(cx)], axis=1) @ M3.T
                keep &= (centers[:,0] >= 0) & (centers[:,0] < outW) & (centers[:,1] >= 0) & (centers[:,1] < outH)

            # AABB from transformed corners, then clip
            x_min = np.clip(rot[:,:,0].min(axis=1), 0, outW-1)
            y_min = np.clip(rot[:,:,1].min(axis=1), 0, outH-1)
            x_max = np.clip(rot[:,:,0].max(axis=1), 0, outW-1)
            y_max = np.clip(rot[:,:,1].max(axis=1), 0, outH-1)
            new_boxes = np.stack([x_min, y_min, x_max, y_max], axis=1)

            # drop tiny/degenerate
            area = (new_boxes[:,2]-new_boxes[:,0]) * (new_boxes[:,3]-new_boxes[:,1])
            keep &= area >= self.min_box_area

            if keep.any():
                return sheared, new_boxes[keep], (labels[keep] if labels is not None else None)

        # fallback
        return image, boxes, labels
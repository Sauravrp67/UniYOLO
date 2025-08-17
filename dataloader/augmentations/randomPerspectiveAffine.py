import cv2
import numpy as np

class RandomPerspectiveAffine:
    def __init__(self, degrees=10, translate=0.1, scale=(0.5,1.5),
                 shear=5, perspective=0.0, p=0.5, pad_canvas=False, fill_value=(123,117,104),
                 min_box_area=4, keep_center=True):
        self.degrees = float(degrees)
        self.translate = float(translate)
        self.scale = scale
        self.shear = float(shear)
        self.persp = float(perspective)
        self.p = float(p)
        self.pad_canvas = bool(pad_canvas)
        self.fill_value = tuple((np.array(fill_value) * 255).astype(np.uint8).tolist())
        self.min_box_area = float(min_box_area)
        self.keep_center = bool(keep_center)

    def __call__(self, image, boxes, labels=None):
        if np.random.rand() > self.p or boxes is None or len(boxes)==0:
            return image, boxes, labels

        H, W = image.shape[:2]
        cx, cy = W/2.0, H/2.0

        # sample params
        ang = np.deg2rad(np.random.uniform(-self.degrees, self.degrees))
        s = np.random.uniform(self.scale[0], self.scale[1])
        shx = np.deg2rad(np.random.uniform(-self.shear, self.shear))
        shy = np.deg2rad(np.random.uniform(-self.shear, self.shear))
        px = np.random.uniform(-self.persp, self.persp) if self.persp>0 else 0.0
        py = np.random.uniform(-self.persp, self.persp) if self.persp>0 else 0.0
        tx = np.random.uniform(-self.translate*W, self.translate*W)
        ty = np.random.uniform(-self.translate*H, self.translate*H)

        # build 3x3
        T1 = np.array([[1,0,-cx],[0,1,-cy],[0,0,1]], np.float32)
        S  = np.array([[s,0,0],[0,s,0],[0,0,1]], np.float32)
        R  = np.array([[np.cos(ang),-np.sin(ang),0],[np.sin(ang),np.cos(ang),0],[0,0,1]], np.float32)
        Sh = np.array([[1,np.tan(shx),0],[np.tan(shy),1,0],[0,0,1]], np.float32)
        P  = np.array([[1,0,0],[0,1,0],[px,py,1]], np.float32)
        T2 = np.array([[1,0,cx],[0,1,cy],[0,0,1]], np.float32)
        Tt = np.array([[1,0,tx],[0,1,ty],[0,0,1]], np.float32)

        M = Tt @ T2 @ P @ Sh @ R @ S @ T1   # final 3x3

        # output size
        if self.pad_canvas:
            # conservative bbox of rotated rectangle (ignore perspective in sizing)
            cos, sin = abs(np.cos(ang))*s, abs(np.sin(ang))*s
            newW = int(W*cos + H*sin); newH = int(W*sin + H*cos)
            outW, outH = max(1,newW), max(1,newH)
        else:
            outW, outH = W, H

        # warp image
        img_warp = cv2.warpPerspective(
            image, M, (outW, outH), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=self.fill_value
        )

        # transform boxes: corners -> homography -> AA box
        b = boxes.astype(np.float32)
        corners = np.stack([
            np.stack([b[:,0], b[:,1]], 1),
            np.stack([b[:,2], b[:,1]], 1),
            np.stack([b[:,2], b[:,3]], 1),
            np.stack([b[:,0], b[:,3]], 1),
        ], 1)  # (N,4,2)
        hom = np.concatenate([corners, np.ones((*corners.shape[:2],1), np.float32)], 2)
        rot = hom @ M.T
        rot_xy = rot[..., :2] / np.clip(rot[..., 2:3], 1e-6, None)

        x_min = np.clip(rot_xy[...,0].min(1), 0, outW-1)
        y_min = np.clip(rot_xy[...,1].min(1), 0, outH-1)
        x_max = np.clip(rot_xy[...,0].max(1), 0, outW-1)
        y_max = np.clip(rot_xy[...,1].max(1), 0, outH-1)
        new_boxes = np.stack([x_min, y_min, x_max, y_max], 1)

        # keep rules
        keep = np.ones(len(new_boxes), dtype=bool)
        if self.keep_center:
            cx0 = (b[:,0]+b[:,2]) * 0.5
            cy0 = (b[:,1]+b[:,3]) * 0.5
            ctr = np.stack([cx0, cy0, np.ones_like(cx0)], 1) @ M.T
            ctr_xy = ctr[:, :2] / np.clip(ctr[:, 2:3], 1e-6, None)
            keep &= (ctr_xy[:,0]>=0)&(ctr_xy[:,0]<outW)&(ctr_xy[:,1]>=0)&(ctr_xy[:,1]<outH)

        area = (new_boxes[:,2]-new_boxes[:,0]) * (new_boxes[:,3]-new_boxes[:,1])
        keep &= area >= self.min_box_area

        new_boxes = new_boxes[keep]
        new_labels = labels[keep] if labels is not None else None
        return img_warp, new_boxes, new_labels

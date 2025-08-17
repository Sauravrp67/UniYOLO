import numpy as np
import cv2

class RandomSampleCrop:
    def __init__(self,verbose = False):
        self.sample_option = (
            None,
            ## Min IOU and Max IOU
            (0.1,None),
            (0.3,None),
            (0.7,None),
            (0.9,None),
            (None,None)

        )
        self.verbose = verbose

    def __call__(self,image,boxes,labels):
        height,width,_ = image.shape

        while True:
            sample_id = np.random.randint(len(self.sample_option))
            sample_mode = self.sample_option[sample_id]
            if sample_mode is None:
                return image,boxes,labels
            
            min_iou,max_iou = sample_mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')
            for _ in range(50):
                current_image = image
                w = np.random.uniform(0.3 * width,width)
                h = np.random.uniform(0.3 * height,height)
                if max(h,w) / min(h,w) > 2:
                    continue

                left = np.random.uniform(width - w)
                top = np.random.uniform(height - h)
                rect = np.array([int(left),int(top),int(left+w),int(top+h)])

                overlap = self.compute_IoU(boxes,rect)
                if min_iou is not None:
                    if overlap.size == 0 or overlap.max() < min_iou:
                        continue

                centers = (boxes[:,:2] + boxes[:,2:]) / 2.0

                m1 = (rect[0] < centers[:,0]) * (rect[1] < centers[:,1])
                m2 = (rect[2] > centers[:,0]) * (rect[3] > centers[:,1])
                mask = m1 * m2
                if not mask.any():
                    continue
                if self.verbose:
                    print(f"Crop_HW:{(h,w)}")
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]
                current_boxes = boxes[mask, :]
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                current_boxes[:, :2] -= rect[:2]
                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                current_boxes[:, 2:] -= rect[:2]
                current_labels = labels[mask]
                return current_image, current_boxes, current_labels
    
    def compute_IoU(self, boxA, boxB):
        inter = self.intersect(boxA, boxB)
        areaA = ((boxA[:, 2]-boxA[:, 0]) * (boxA[:, 3]-boxA[:, 1]))
        areaB = ((boxB[2]-boxB[0]) * (boxB[3]-boxB[1]))
        union = areaA + areaB - inter
        return inter / union

    def intersect(self, boxA, boxB):
        max_xy = np.minimum(boxA[:, 2:], boxB[2:])
        min_xy = np.maximum(boxA[:, :2], boxB[:2])
        inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
        return inter[:, 0] * inter[:, 1]
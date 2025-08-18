import numpy as np
import cv2

class LetterBox:
    # All transformation class has input in numpy formats
    def __init__(self,new_shape = (448,448),color= (0,0,0)):
        self.new_shape = (new_shape,new_shape) if isinstance(new_shape,int) else new_shape
        self.color = color

    def __call__(self,image,boxes,labels = None):
        shape = image.shape[:2] #(height,width)
        #Scaling ratio, we select the min of new/old so that bigger dimension maps to the max of desire dimension
        r = min(self.new_shape[0]/shape[0],self.new_shape[1]/shape[1])
        #Compute how much padding is required
        new_unpad = int(round(shape[1] * r)),int(round(shape[0] * r)) #(width,height)
        dw,dh = self.new_shape[1] - new_unpad[0],self.new_shape[0] - new_unpad[1]
        #We need to equally divide these padding required into two sides 
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            image = cv2.resize(image,new_unpad,interpolation = cv2.INTER_LINEAR)
        top,bottom = int(round(dh - 0.1)) , int(round(dh + 0.1))
        left,right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image,top,bottom,left,right,cv2.BORDER_CONSTANT,value=self.color)

        if boxes is None:
            return image,boxes,labels
        elif not len(boxes) > 0:
            return image,boxes,labels
        else:
            boxes[:,:2] = (boxes[:,:2] * (new_unpad[0],new_unpad[1])) + (left,top)
            boxes[:,:2] /= (image.shape[1],image.shape[0])
            boxes[:,2:] /= (image.shape[1] / new_unpad[0],image.shape[0] / new_unpad[1])
        return image,boxes,labels
    
import numpy as np
import cv2

import numpy as np
import cv2

class UnletterBox:
    """
    Inverse of your LetterBox:
      - Input `image`  : letterboxed image (HxWxC) produced with LetterBox(new_shape)
      - Input `boxes`  : normalized to the letterboxed canvas, format [cx, cy, w, h]
      - Input `orig_shape`: (H0, W0) of the pre-letterbox image

    Returns:
      - unletter_img : image cropped (pads removed) and resized back to (H0, W0)
      - boxes_orig   : boxes normalized to the ORIGINAL image size (same [cx,cy,w,h] format)
      - labels       : passthrough
    """
    def __init__(self, new_shape=(448, 448)):
        self.new_shape = (new_shape, new_shape) if isinstance(new_shape, int) else tuple(new_shape)

    def __call__(self, image, boxes=None, labels=None, orig_shape=None):
        assert orig_shape is not None, "orig_shape=(H0,W0) of the pre-letterbox image is required."
        H0, W0 = orig_shape                 
        H,  W  = image.shape[:2]            

        r = min(self.new_shape[0] / H0, self.new_shape[1] / W0)
        new_unpad = (int(round(W0 * r)), int(round(H0 * r)))  
        dw, dh = self.new_shape[1] - new_unpad[0], self.new_shape[0] - new_unpad[1]
        dw *= 0.5
        dh *= 0.5
        top, bottom  = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right  = int(round(dw - 0.1)), int(round(dw + 0.1))

        crop = image[top:H-bottom, left:W-right]
        unletter_img = cv2.resize(crop, (W0, H0), interpolation=cv2.INTER_LINEAR)


        boxes_orig = boxes
        if boxes is not None and len(boxes) > 0:
            out = boxes.astype(np.float32).copy()

            out[:, :2] = (out[:, :2] * (W, H) - (left, top)) / (new_unpad[0], new_unpad[1])

            out[:, 2:] = out[:, 2:] * (W / new_unpad[0], H / new_unpad[1])

            np.clip(out, 0.0, 1.0, out=out)
            boxes_orig = out

        return unletter_img, boxes_orig, labels


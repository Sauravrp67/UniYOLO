import numpy as np
import cv2

class Compose:
    def __init__(self,transforms):
        self.transforms = transforms
    
    def __call__(self,image,boxes = None,labels = None):
        for tf in self.transforms:
            if tf is None:
                continue
            else:
                image,boxes,labels = tf(image,boxes,labels)
        return image,boxes,labels
    
class Normalize:
    def __init__(self,mean,std):
        self.mean = np.array(mean,dtype = np.float32)
        self.std = np.array(std,dtype = np.float32)
    
    def __call__(self,image,boxes = None,labels = None):
        image /= 255
        image -= self.mean
        image /= self.std
        return image,boxes,labels

# class Normalize:
#     """Same math: x = (x/255 - mean)/std; do it in-place with OpenCV."""
#     def __init__(self, mean, std):
#         self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
#         self.inv255 = 1.0 / 255.0
#         self.inv_std = (1.0 / np.array(std, dtype=np.float32)).reshape(1, 1, 3)
#         # Optional reusable float32 buffer
#         self.buf = None

#     def __call__(self, image_u8, boxes=None, labels=None):
#         # Ensure uint8 input
#         assert image_u8.dtype == np.uint8
#         if self.buf is None or self.buf.shape != image_u8.shape:
#             self.buf = np.empty_like(image_u8, dtype=np.float32)

#         img = self.buf  # float32 buffer
#         img[...] = image_u8  # upcast copy: uint8 -> float32

#         # In-place normalize with NumPy (no OpenCV binding issues)
#         np.multiply(img, self.inv255, out=img)   # img *= 1/255
#         np.subtract(img, self.mean,  out=img)    # img -= mean
#         np.multiply(img, self.inv_std, out=img)  # img *= 1/std

#         return img, boxes, labels
class Resize:
    def __init__(self,size = 640):
        self.size = size

    def __call__(self,image,boxes = None,labels = None):
        image = cv2.resize(image,(self.size,self.size))
        return image,boxes,labels
    

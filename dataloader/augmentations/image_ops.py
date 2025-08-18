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
    
class Resize:
    def __init__(self,size = 640):
        self.size = size

    def __call__(self,image,boxes = None,labels = None):
        image = cv2.resize(image,(self.size,self.size))
        return image,boxes,labels
    

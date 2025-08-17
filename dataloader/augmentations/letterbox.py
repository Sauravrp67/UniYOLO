import numpy as np
import cv2

class LetterBox:
    # All transformation class has input in numpy format
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
        boxes[:,:2] = (boxes[:,:2] * (new_unpad[0],new_unpad[1])) + (left,top)
        boxes[:,:2] /= (image.shape[1],image.shape[0])
        boxes[:,2:] /= (image.shape[1] / new_unpad[0],image.shape[0] / new_unpad[1])
        return image,boxes,labels
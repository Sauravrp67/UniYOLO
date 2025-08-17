import numpy as np


class ToXminYminXmaxYmax:
    def __call__(self,image,boxes,labels = None):
        x1y1 = boxes[:,:2] - boxes[:,2:] / 2
        x2y2 = boxes[:,:2] + boxes[:,2:] / 2
        boxes = np.concatenate((x1y1,x2y2),axis = 1).clip(min = 0,max = 1)
        return image,boxes,labels

class ToAbsoluteCoords:
    def __call__(self,image,boxes,labels = None):
        H,W,_ = image.shape
        boxes[:,0::2]  *= W
        boxes[:,1::2] *= H

        return image,boxes,labels

class ToPercentCoords:
    def __call__(self,image,boxes,labels = None):
        H,W,_ = image.shape
        boxes[:,0::2] /= W
        boxes[:,1::2] /= H
        return image,boxes,labels
    
class ToXcenYcenWH:
    def __call__(self,image,boxes,labels = None):
        CxCy = (boxes[:,:2] + boxes[:,2:]) / 2
        WH = (boxes[:,2:] - boxes[:,:2])
        boxes = np.concatenate((CxCy,WH),axis = 1).clip(min = 0,max = 1)
        return image,boxes,labels 
    
import numpy as np

class HorizontalFlip:
    def __call__(self, image, boxes, labels=None):
        _, width, _ = image.shape
        if np.random.randint(2):
            image = image[:, ::-1, :]
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, labels
    
class VerticalFlip:
    def __call__(self,image,boxes,labels):
        height,_,_ = image.shape
        if np.random.randint(2):
            image = image[::-1, :,:]
            boxes[:,1::2] = height - boxes[:,-1::-2]
        return image,boxes,labels
    
class Expand:
    def __init__(self,mean,verbose = False):
        self.mean = np.array(mean,dtype=np.float32) * 255.0
        self.verbose = verbose
    def __call__(self,image,boxes,labels = None):
        condition = np.random.randint(2)
        if condition:
            return image,boxes,labels
        height,width,channel = image.shape
        scale = np.random.uniform(1,2)
        scale = scale
        left = np.random.uniform(0,width*scale - width)
        top = np.random.uniform(0,height * scale - height)

        expand_image = np.zeros((int(height * scale),int(width * scale),channel),dtype = image.dtype)
        expand_image[...] = self.mean
        expand_image[int(top):int(top+height),int(left):int(left + width), :] = image
        boxes[:,:2] += (int(left),int(top))
        boxes[:,2:] += (int(left),int(top))
        if self.verbose:
            print(f"Expand:{not bool(condition)}, Scale:{scale}")
        return expand_image,boxes,labels
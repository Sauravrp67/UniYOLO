import cv2
import numpy as np
from collections import namedtuple

MEAN = 0.485, 0.456, 0.406 # RGB
STD = 0.229, 0.224, 0.225 # RGB


class RandomBrightness:
    def __init__(self,delta = 32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self,image,boxes = None,labels = None):
        condition = np.random.randint(2)
        if condition:
            value = np.random.uniform(-self.delta,self.delta)
            image += value
        return image,boxes,labels
    
class RandomContrast:
    def __init__(self,lower = 0.5,upper = 1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "Constrast Upper must be >= lower."
        assert self.lower >= 0, "constrast lower must be non-negative"

    def __call__(self,image,boxes = None,labels = None):
        condition = np.random.randint(2)
        if condition:
            value= np.random.uniform(self.lower,self.upper)
            image *= value
        return image,boxes,labels
    
class ConvertColor:
    def __init__(self,color_from = "RGB",color_to = "HSV"):
        self.color_from = color_from
        self.color_to = color_to

    def __call__(self,image,boxes = None,labels = None):
        if self.color_from == "RGB" and self.color_to == "HSV":
            image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        elif self.color_from == "HSV" and self.color_to == "RGB":
            image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image,boxes,labels
class RandomHue:
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        condition = np.random.randint(2)
        if condition:
            value = np.random.uniform(-self.delta,self.delta)
            image[:, :, 0] += value
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomSaturation:
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        condition = np.random.randint(2)
        if condition:
            value = np.random.uniform(self.lower, self.upper)
            image[:, :, 1] *= value
        return image, boxes, labels
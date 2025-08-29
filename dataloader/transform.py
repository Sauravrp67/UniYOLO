import numpy as np
from collections import namedtuple
from augmentations import *
from augmentations.sampler import SampleProvider


MEAN = 0.485, 0.456, 0.406 # RGB
STD = 0.229, 0.224, 0.225 # RGB


class SelectOne:
    """
    Picks one transform from a list(or none) per sample.
    Each candidate must be a callable object:(image,boxes,labels) -> (image,boxes,labels)

    Example:
        SelectOne(
        [mixup_tf,cutmix_tf,None],
        probs = (0.12,0.10,0.78) #Must sum to 1.0
        )
    """
    def __init__(self,transforms,probs):
        assert len(transforms) == len(probs), "transform/probs length mismatch"
        self.transforms = transforms
        self.probs = np.asarray(probs,dtype = np.float64)
        self.probs /= self.probs.sum() #normalize just in case

    def __call__(self,image,boxes = None,labels = None):
        idx = np.random.choice(len(self.transforms),p = self.probs)
        tf = self.transforms[idx]
        if tf is None:
            return image,boxes,labels
        return tf(image,boxes,labels)

class BasicTransform:
    def __init__(self,input_size,mean =MEAN,std = STD):
        mean = np.array(mean,dtype = np.float32)
        std = np.array(std,dtype = np.float32)
        self.tfs = Compose(
            [
                LetterBox(new_shape = input_size),
                Normalize(mean = mean,std = std)
            ]
        )
    def __call__(self,image,boxes = None,labels = None):
        image = image.astype(np.float32)
        image,boxes,labels = self.tfs(image,boxes,labels)
        return image,boxes,labels

class AugmentTransform:
    def __init__(self,input_size,dataset,mean = MEAN,std = STD,cutmix_mixup:bool = False,mosaic = True):
        mean = np.array(mean,dtype = np.float32)
        std = np.array(std,dtype = np.float32)

        photo_metric_transforms = [
            RandomBrightness(), 
            RandomContrast(),
            ConvertColor(color_from="RGB", color_to="HSV"),
            RandomHue(),
            RandomSaturation(),
            ConvertColor(color_from="HSV", color_to="RGB"),
        ]

        pre_geometric_transforms = [
            ToXminYminXmaxYmax(),
            ToAbsoluteCoords()
        ]
        multi = []
        multi_image_augment = []
        
        if mosaic:
            multi.append(
                RandomMosaic(
                provider = SampleProvider(dataset=dataset),
                output_size = 416,
                pad_value = mean,
                scale_range=(0.9,1.1),
                center_ratio_range=(1.0,1.0),
                enforce_min_fill=(0.75),
                min_visible=0.30,
                p = 0.5
                )
                )

        if cutmix_mixup:
            cutmix_provider = SampleProvider(dataset=dataset)
            multi_image_augment.append(RandomCutMix(provider = cutmix_provider,
                                                    min_ratio = 0.3,
                                                    max_ratio=0.5,
                                                    occ_thresh = 0.5,
                                                    p = 0.5))
            mixup_provider = SampleProvider(dataset = dataset)
            multi_image_augment.append(RandomMixUp(provider = mixup_provider,
                                                   pad_value = mean,
                                                   p = 0.8,
                                                   min_visible_lambda = 0.1,
                                                   alpha = 0.4))
            
            multi_image_augment.append(None)
            multi.append(
                SelectOne(
                transforms=multi_image_augment,
                probs=[0.2,0.2,0.6]
                )
                )
        geometric_transforms = [

            # Expand(mean = mean,verbose = False),
            # RandomSampleCrop(verbose = False),
            RandomPerspectiveAffine(fill_value = mean,perspective=0.0005),
            HorizontalFlip(),
            CustomCutout(
                fill_value = 0,
                bbox_removal_threshold=0.5,
                min_cutout_pct=5,
                max_cutout_pct = 30,
                p = 0.5,
                min_holes = 3,
                max_holes=7,
                verbose=False
            ),
            ]
        post_geometric_transforms = [
                ToPercentCoords(),
                ToXcenYcenWH(),
                LetterBox(new_shape = input_size),
                Normalize(mean = mean,std = std)
        ]

        transforms = photo_metric_transforms + pre_geometric_transforms + multi + geometric_transforms + post_geometric_transforms
        self.tfs = Compose(transforms = transforms)

    def __call__(self,image,boxes = None,labels = None):
        image = image.astype(np.float32)
        image,boxes,labels = self.tfs(image,boxes,labels)
        return image,boxes,labels


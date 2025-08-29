import torch
import numpy as np
import cv2
from dataloader import BasicTransform,ToXcenYcenWH,ToXminYminXmaxYmax,ToAbsoluteCoords,UnletterBox
from utils import denormalize,to_tensor,transform_xcycwh_to_x1y1x2y2,filter_confidence,run_NMS,transform_x1y1x2y2_to_xcycwh,scale_to_original,to_image
from model import YOLOv3
import time
from dataloader import Dataset
from torch.utils.data import DataLoader



VOC_CLASSES = {
    0:'aeroplane', 1:'bicycle', 2:'bird', 3:'boat', 4:'bottle',
    5:'bus', 6:'car', 7:'cat', 8:'chair', 9:'cow',
    10:'diningtable', 11:'dog', 12:'horse', 13:'motorbike', 14:'person',
    15:'pottedplant', 16:'sheep', 17:'sofa', 18:'train', 19:'tvmonitor'
}

def _overlaps(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)

def _place_label_avoiding_collisions(x1, y1, x2, y2, text_size, img_w, img_h, occupied, pad=3, step=4):
    tw, th, bl = text_size
    # candidate rectangles: above, inside top, below
    cands = []
    # above
    tx1 = x1
    ty2 = y1
    ty1 = max(0, ty2 - th - bl - 2*pad)
    tx2 = min(img_w-1, tx1 + tw + 2*pad)
    cands.append([tx1, ty1, tx2, ty2])
    # inside top
    tx1 = x1
    ty1 = y1 + 1
    ty2 = min(img_h-1, ty1 + th + bl + 2*pad)
    tx2 = min(img_w-1, tx1 + tw + 2*pad)
    cands.append([tx1, ty1, tx2, ty2])
    # below
    tx1 = x1
    ty1 = min(img_h-1, y2 + 1)
    ty2 = min(img_h-1, ty1 + th + bl + 2*pad)
    tx2 = min(img_w-1, tx1 + tw + 2*pad)
    cands.append([tx1, ty1, tx2, ty2])

    # try candidates, then nudge down until free
    for rect in cands:
        # clamp horizontally
        shift_x = max(0, rect[2] - (img_w - 1))
        rect[0] -= shift_x; rect[2] -= shift_x
        if not any(_overlaps(rect, r) for r in occupied):
            return rect
        # nudge vertically if colliding
        tries = 0
        while tries < 30:
            rect[1] = min(img_h-1, rect[1] + th + bl + step)
            rect[3] = min(img_h-1, rect[1] + th + bl + 2*pad)
            if rect[3] >= img_h: break
            if not any(_overlaps(rect, r) for r in occupied):
                return rect
            tries += 1
    # fallback: return first candidate even if overlapping
    return cands[0]

def visualize_image(image, boxes, labels=None, class_map=VOC_CLASSES,
                    to_XYXY=False, to_abs=False, color=(0,0,255)):
    img = np.ascontiguousarray(image)
    h, w = img.shape[:2]
    boxes_temp = boxes

    if to_XYXY:
        boxes_temp = ToXminYminXmaxYmax()(img, boxes_temp)[1]
    if to_abs:
        boxes_temp = ToAbsoluteCoords()(img, boxes_temp)[1]

    boxes_xyxy = boxes_temp.astype(np.int32)
    boxes_xyxy[:, [0,2]] = np.clip(boxes_xyxy[:, [0,2]], 0, w-1)
    boxes_xyxy[:, [1,3]] = np.clip(boxes_xyxy[:, [1,3]], 0, h-1)

    if labels is None:
        labels = [None] * len(boxes_xyxy)
    if class_map is None:
        class_map = {0:'aeroplane',1:'bicycle',2:'bird',3:'boat',4:'bottle',
                     5:'bus',6:'car',7:'cat',8:'chair',9:'cow',10:'diningtable',
                     11:'dog',12:'horse',13:'motorbike',14:'person',15:'pottedplant',
                     16:'sheep',17:'sofa',18:'train',19:'tvmonitor'}

    font = cv2.FONT_HERSHEY_SIMPLEX
    base_scale = max(0.45, min(1.2, w/640.0*0.6))
    thickness = max(1, int(round(base_scale + 0.5)))
    occupied = []  # placed label rectangles to avoid

    # draw boxes first
    for (x1,y1,x2,y2) in boxes_xyxy:
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)

    # draw labels with collision avoidance
    for (x1,y1,x2,y2), cls in zip(boxes_xyxy, labels):
        if cls is None:
            continue
        name = class_map.get(int(cls), str(cls))
        (tw, th), bl = cv2.getTextSize(name, font, base_scale, thickness)
        rect = _place_label_avoiding_collisions(x1, y1, x2, y2, (tw, th, bl), w, h, occupied, pad=3)
        occupied.append(rect)

        rx1, ry1, rx2, ry2 = rect
        # background
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), color, -1)
        # text baseline
        ty = min(h-1, ry1 + th + 1)
        cv2.putText(img, name, (rx1 + 3, ty), font, base_scale, (255,255,255), thickness, cv2.LINE_AA)

    return img

def image_preprocess(image:np.ndarray,tf):
    input_image_np,_,_ = tf(image,boxes = None,labels = None)
    input_image = to_tensor(input_image_np).unsqueeze(0)

    return input_image,input_image_np


def post_process(image_letterboxed:np.ndarray,image_original_shape:tuple,predictions:np.ndarray,tf:UnletterBox,letterboxed:bool = False):
    predictions[:,1:5] = transform_xcycwh_to_x1y1x2y2(boxes = predictions[:,1:5],clip_max = 1.0)
    predictions = filter_confidence(predictions,conf_threshold=0.15)
    predictions = run_NMS(predictions,iou_threshold=0.5)

    boxes = predictions[:,1:5].copy()
    if letterboxed:
        boxes *= image_letterboxed.shape[0]
        image_out = image_letterboxed
    else:
        boxes = transform_x1y1x2y2_to_xcycwh(predictions[:,1:5]).copy()
        image_transformed,boxes,_ = tf(image_letterboxed,boxes = boxes,labels = None,orig_shape=(image_original_shape[0],image_original_shape[1]))
        boxes = transform_xcycwh_to_x1y1x2y2(boxes,clip_max=1.0)
        boxes = scale_to_original(boxes,scale_w = image_transformed.shape[1],scale_h = image_transformed.shape[0])
        image_out = image_transformed
    label = predictions[:,0]
    conf = predictions[:,-1]

    return image_out,boxes,label,conf


anchors = [[0.248,      0.7237237 ],
        [0.36144578, 0.53      ],
        [0.42,       0.9306667 ],
        [0.456,      0.6858006 ],
        [0.488,      0.8168168 ],
        [0.6636637,  0.274     ],
        [0.806,      0.648     ],
        [0.8605263,  0.8736842 ],
        [0.944,      0.5733333 ]]


val_dataset = Dataset('/workspace/data/voc_vitis.yaml',phase = 'val')

transform = BasicTransform(input_size = 416)
val_dataset.load_transformer(transformer = transform)
val_loader = DataLoader(val_dataset,batch_size = 8,shuffle = True,num_workers = 1,collate_fn = Dataset.collate_fn,pin_memory = True,)

minibatch = next(iter(val_loader))

input_image = minibatch[1]
shape = minibatch[-1]

print(input_image.shape)
print(shape)
# input_image,input_image_np = image_preprocess(image,transform)

input_size = 416
model = YOLOv3(input_size = input_size, num_classes = 20, anchors = anchors, model_type = 'base', pretrained = True).to('cpu')

model.eval()
model.zero_grad()
model.set_grid_xy(input_size = 416)

start_time = time.time()
predictions = model(input_image)
predictions = predictions.detach()
print(f"Model processing time{time.time() - start_time}")

unletterboxing = UnletterBox(new_shape = (416,416))

for image_tensor,predictions,original_shape in zip(input_image,predictions,shape):
    
    input_image_np = to_image(image_tensor)
    image,boxes,label,conf = post_process(input_image_np,image_original_shape=original_shape,predictions = predictions.numpy(),tf = unletterboxing,letterboxed=False)
    img = visualize_image(image = image,boxes = boxes,labels = label)
    cv2.imshow("Prediction",img)
    cv2.waitKey(0)

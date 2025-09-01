import numpy as np
import cv2
import math

import os
import cv2
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


# from dataloader.transform import UnletterBox

MEAN = 0.485, 0.456, 0.406 # RGB
STD = 0.229, 0.<224, 0.225 # RGB

CLASS_INFO = {
    0:'aeroplane',1:'bicycle',2:'bird',3:'boat',4:'bottle',5:'bus',6:'car',7:'cat',
    8:'chair',9:'cow',10:'diningtable',11:'dog',12:'horse',13:'motorbike',14:'person',
    15:'pottedplant',16:'sheep',17:'sofa',18:'train',19:'tvmonitor'
}

def denormalize(image, mean=MEAN, std=STD):
    image_temp = image.copy()
    image_temp *= std
    image_temp += mean
    image_temp *= 255.
    return image_temp.astype(np.uint8)

def clip_xyxy(boxes:np.ndarray,W:int,H:int) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    
    boxes[:,[0,2]] = np.clip(boxes[:,[0,2]],0,max(0,W - 1))
    boxes[:,[1,3]]= np.clip(boxes[:,[1,3]],0,max(0,H - 1))
    return boxes


def transform_xcycwh_to_x1y1wh(boxes):
    x1y1 = boxes[:, :2] - boxes[:, 2:] / 2
    wh = boxes[:, 2:]
    return np.concatenate((x1y1, wh), axis=1).clip(min=0)

def transform_xcycwh_to_x1y1x2y2(boxes, clip_max=None):
    x1y1 = boxes[:, :2] - boxes[:, 2:] / 2
    x2y2 = boxes[:, :2] + boxes[:, 2:] / 2
    x1y1x2y2 = np.concatenate((x1y1, x2y2), axis=1)
    return x1y1x2y2.clip(min=0, max=clip_max if clip_max is not None else 1)

def transform_x1y1x2y2_to_x1y1wh(boxes):
    x1y1 = boxes[:, :2]
    wh = boxes[:, 2:] - boxes[:, :2]
    return np.concatenate((x1y1, wh), axis=1)

def transform_x1y1x2y2_to_xcycwh(boxes):
    wh = boxes[:, 2:] - boxes[:, :2]
    xcyc = boxes[:, :2] + wh / 2
    return np.concatenate((xcyc, wh), axis=1)

def scale_to_original(boxes,scale_w,scale_h):
    boxes[:,[0,2]] *= scale_w
    boxes[:,[1,3]] *= scale_h
    return boxes.round(2)

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
    
def scale_coords(img1_shape, coords, img0_shape):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    coords = scale_to_original(boxes=coords, scale_w=img1_shape[1], scale_h=img1_shape[0])
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def scale_to_original(boxes, scale_w, scale_h):
    boxes[:,[0,2]] *= scale_w
    boxes[:,[1,3]] *= scale_h
    return boxes.round(2)

def scale_to_norm(boxes, image_w, image_h):
    boxes[:,[0,2]] /= image_w
    boxes[:,[1,3]] /= image_h
    return boxes

def filter_confidence(prediction, conf_threshold=0.01):
    keep = (prediction[:, 0] > conf_threshold)
    conf = prediction[:, 0][keep]
    box = prediction[:, 1:5][keep]
    cls_id = prediction[:, 5][keep]
    return np.concatenate([cls_id[:, np.newaxis], box, conf[:, np.newaxis]], axis=-1)


def hard_NMS(boxes, scores, iou_threshold):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []                                             
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(1e-10, xx2 - xx1)
        h = np.maximum(1e-10, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def run_NMS(prediction, iou_threshold, maxDets=100):
    keep = np.zeros(len(prediction), dtype=np.int32)
    for cls_id in np.unique(prediction[:, 0]):
        inds = np.where(prediction[:, 0] == cls_id)[0]
        if len(inds) == 0:
            continue
        cls_boxes = prediction[inds, 1:5]
        cls_scores = prediction[inds, 5]
        cls_keep = hard_NMS(boxes=cls_boxes, scores=cls_scores, iou_threshold=iou_threshold)
        keep[inds[cls_keep]] = 1
    prediction = prediction[np.where(keep > 0)]
    order = prediction[:, 5].argsort()[::-1]
    return prediction[order[:maxDets]]

def imwrite(filename, img):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False

def preprocess(image, tf):
    """
    Basic resize/normalize. Adapt to your training pipeline.
    """
    input_image_np,_,_ = tf(image,boxes = None,labels = None)
    return input_image_np

def post_process(image_letterboxed:np.ndarray,image_original:np.ndarray,predictions:np.ndarray,tf,letterboxed:bool = False,conf_thresh:float = 0.2,nms_iou_thresh:float = 0.5):
    predictions[:,1:5] = transform_xcycwh_to_x1y1x2y2(boxes = predictions[:,1:5],clip_max = 1.0)
    predictions = filter_confidence(predictions,conf_threshold=conf_thresh)
    predictions = run_NMS(predictions,iou_threshold=nms_iou_thresh)
    
    boxes = predictions[:,1:5].copy()
    if letterboxed:
        boxes *= image_letterboxed.shape[0]
        image_out = image_letterboxed
    else:
        boxes = transform_x1y1x2y2_to_xcycwh(predictions[:,1:5]).copy()
        image_transformed,boxes,_ = tf(image_letterboxed,boxes = boxes,labels = None,orig_shape=(image_original.shape[0],image_original.shape[1]))
        boxes = transform_xcycwh_to_x1y1x2y2(boxes,clip_max=1.0)
        boxes = scale_to_original(boxes,scale_w = image_transformed.shape[1],scale_h = image_transformed.shape[0])
        image_out = image_transformed
    label = predictions[:,0]
    conf = predictions[:,-1]

    return image_out,boxes,label,conf

def id2name(ids):
    return [CLASS_INFO.get(int(i), str(int(i))) for i in ids]

def draw_dets(img, boxes_xyxy, labels, conf):
    img = img.copy()
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return img
    for (x1, y1, x2, y2), lid, sc in zip(boxes_xyxy.astype(int), labels.astype(int), conf):
        name = CLASS_INFO.get(int(lid), str(int(lid)))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f"{name} {sc:.2f}", (x1, max(0, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return img

def hardsigmoid_np(x: np.ndarray) -> np.ndarray:
    # PyTorch HardSigmoid: clamp(x/6 + 0.5, 0, 1)
    y = x / 6.0 + 0.5
    out = np.empty_like(x, dtype=x.dtype)
    np.clip(y, 0.0, 1.0, out=out)
    return out

def sigmoid_np(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    out = np.empty_like(x, dtype=x.dtype)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out

class BoxDecoderNP:
    """
    NumPy decoder matching:
        pred_obj       = Hardsigmoid(out[..., :1])
        pred_box_txty  = Hardsigmoid(out[..., 1:3])
        pred_box_twth  = out[..., 3:5]
        pred_cls       = out[..., 5:]
        pred_score     = pred_obj * sigmoid(pred_cls)
        pred_box       = decode(tx,ty,tw,th) -> (xc, yc, w, h)
        return concat(score, box, label).reshape(B, N*A, 6)
    """
    def __init__(self, predictions: np.ndarray, anchors: np.ndarray):
    
        assert predictions.ndim == 4, f"Expected (B,N,A,D), got {predictions.shape}"
        assert anchors.ndim == 2 and anchors.shape[1] == 2, f"Expected (A,2), got {anchors.shape}"

        self.out = predictions.astype(np.float32, copy=False)
        self.anchors = anchors.astype(np.float32, copy=False)
        self.B, self.N, self.A, self.D = self.out.shape

        S = int(round(math.sqrt(self.N)))
        if S * S != self.N:
            raise ValueError(f"N={self.N} is not a perfect square; cannot infer grid size")
        self.grid_size = S

        gy, gx = np.meshgrid(np.arange(S, dtype=np.float32),
                             np.arange(S, dtype=np.float32),
                             indexing="ij")                 
        gx = gx.reshape(1, -1, 1)                           
        gy = gy.reshape(1, -1, 1)                           
        self.grid_x = gx
        self.grid_y = gy

        self.aw = self.anchors[:, 0].reshape(1, 1, self.A)  
        self.ah = self.anchors[:, 1].reshape(1, 1, self.A)  

    def decode_predictions(self) -> np.ndarray:
        # Slices
        obj_logits  = self.out[..., :1]          
        txty_logits = self.out[..., 1:3]         
        twth        = self.out[..., 3:5]         
        cls_logits  = self.out[..., 5:]          

        # Activations
        pred_obj      = hardsigmoid_np(obj_logits)     
        pred_box_txty = hardsigmoid_np(txty_logits)    
        pred_cls_sig  = sigmoid_np(cls_logits)         
        # Decode bbox
        pred_box = self._transform_pred_box(np.concatenate([pred_box_txty, twth], axis=-1))  

        # Scores and labels
        pred_score_full = pred_obj * pred_cls_sig            
        pred_score = pred_score_full.max(axis=-1)            
        pred_label = pred_score_full.argmax(axis=-1).astype(np.int64)  

        pred_out = np.concatenate(
            [
                pred_score[..., None],   
                pred_box,                 
                pred_label[..., None],    
            ],
            axis=-1
        )

        # Flatten grid*anchors like PyTorch's flatten(1,2): (B, N*A, 6)
        return pred_out.reshape(self.B, self.N * self.A, pred_out.shape[-1])

    def _transform_pred_box(self, pred_box: np.ndarray) -> np.ndarray:
        tx = pred_box[..., 0]                           # (B,N,A)
        ty = pred_box[..., 1]                           # (B,N,A)
        tw = pred_box[..., 2]                           # (B,N,A)
        th = pred_box[..., 3]                           # (B,N,A)

        # Centers: (tx + grid) / S   ; grid_x/y shaped (1,N,1) broadcast over B and A
        S = float(self.grid_size)
        xc = (tx + self.grid_x) / S                     # (B,N,A) via broadcast
        yc = (ty + self.grid_y) / S

        # Sizes: exp(tw/th) * anchor_w/h
        # clip logits a bit to avoid overflow in exp
        tw_clipped = np.clip(tw, -20.0, 20.0)
        th_clipped = np.clip(th, -20.0, 20.0)
        w = np.exp(tw_clipped) * self.aw                # (B,N,A)
        h = np.exp(th_clipped) * self.ah

        # Stack to (B,N,A,4)
        return np.stack([xc, yc, w, h], axis=-1).astype(np.float32, copy=False)

def concat_heads_flatten_anchors(t0, t1, t2):
    B, A, D = t0.shape[0], t0.shape[2], t0.shape[3]
    def flat(x): return x.reshape(B, -1, D)  # (B, S*A, 25)
    return np.concatenate([flat(t0), flat(t1), flat(t2)], axis=1)
#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import cv2
import torch
from utils import denormalize,to_tensor,transform_xcycwh_to_x1y1x2y2,filter_confidence,run_NMS,transform_x1y1x2y2_to_xcycwh,scale_to_original,to_image
import numpy as np
from dataloader import BasicTransform,UnletterBox
# -------------------------------------------------------------------
# Import YOLO model families
# You can extend this with more (V4, V5, V6...) under model/
# -------------------------------------------------------------------
from model import YOLOv3
# from model.V4.yolo import YOLOv4
# from model.V5.yolo import YOLOv5


def load_model(version: str, pretrained:bool, device: str,input_size: int, num_classes: int,model_type: str,anchors):
    """
    Initialize YOLO model and load checkpoint.
    """
    if version.upper() == "V3":
        model = YOLOv3(input_size = input_size, num_classes = num_classes, anchors = anchors, model_type = model_type, pretrained = True).to(device)
    # elif version.upper() == "V4":
    #     from model.V4.yolo import YOLOv4
    #     model = YOLOv4()
    # elif version.upper() == "V5":
    #     from model.V5.yolo import YOLOv5
    #     model = YOLOv5()
    else:
        raise ValueError(f"Unsupported model version: {version}")

    model.to(device).eval()
    model.zero_grad()
    model.set_grid_xy(input_size = input_size)
    return model


def preprocess(image, tf):
    """
    Basic resize/normalize. Adapt to your training pipeline.
    """
    input_image_np,_,_ = tf(image,boxes = None,labels = None)
    input_image = to_tensor(input_image_np).unsqueeze(0)
    return input_image


def post_process(image_letterboxed:np.ndarray,image_original:np.ndarray,predictions:np.ndarray,tf:UnletterBox,letterboxed:bool = False,conf_thresh:float = 0.2,nms_iou_thresh:float = 0.5):
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


CLASS_INFO = {
    0:'aeroplane',1:'bicycle',2:'bird',3:'boat',4:'bottle',5:'bus',6:'car',7:'cat',
    8:'chair',9:'cow',10:'diningtable',11:'dog',12:'horse',13:'motorbike',14:'person',
    15:'pottedplant',16:'sheep',17:'sofa',18:'train',19:'tvmonitor'
}

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


def run_inference(model, device, source, input_size, conf_thresh, nms_iou_thresh):
    tf_pre  = BasicTransform(input_size=input_size)
    tf_unlb = UnletterBox(new_shape=input_size)

    # ----- image -----
    if Path(source).suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"]:
        img = cv2.imread(source)
        t0 = time.time()
        image_tensor = preprocess(img, tf=tf_pre).to(device)
        preds = model(image_tensor)
        image_np = to_image(image_tensor)  # letterboxed canvas as numpy (H,W,C)
        image_out, boxes, labels, conf = post_process(
            image_np, img, predictions=preds.detach().cpu().numpy(),
            tf=tf_unlb, conf_thresh=conf_thresh, nms_iou_thresh=nms_iou_thresh,letterboxed=False
        )
        fps = 1.0 / (time.time() - t0)
        names = id2name(labels) if labels is not None and len(labels) else []
        print(f"FPS: {fps:.2f} | labels: {', '.join(sorted(set(names)))}" if names else f"FPS: {fps:.2f} | labels: none")

        vis = draw_dets(image_out, boxes, labels, conf)
        cv2.imshow("Prediction", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # ----- video / webcam -----
    cap = cv2.VideoCapture(0 if source == "webcam" else source)
    assert cap.isOpened(), f"Failed to open {source}"
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.time()
        image_tensor = preprocess(frame, tf=tf_pre).to(device)
        preds = model(image_tensor)
        image_np = to_image(image_tensor.squeeze(0))
        image_out, boxes, labels, conf = post_process(
            image_np, frame, predictions=preds.squeeze(0).detach().numpy(),
            tf=tf_unlb, conf_thresh=conf_thresh, nms_iou_thresh=nms_iou_thresh,letterboxed=False
        )
        fps = 1.0 / (time.time() - t0)
        names = id2name(labels) if labels is not None and len(labels) else []
        print(f"FPS: {fps:.2f} | labels: {', '.join(sorted(set(names)))}" if names else f"FPS: {fps:.2f} | labels: none")

        vis = draw_dets(image_out, boxes, labels, conf)
        cv2.putText(vis, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Prediction", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()



# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser("YOLO Inference")
    parser.add_argument("--pretrained", type=bool, required=True,
                        help="Path to .pt checkpoint")
    parser.add_argument("--source", type=str, required=True,
                        help="Source: image.jpg | video.mp4 | webcam")
    parser.add_argument("--model-version", type=str, default="V3",
                        help="YOLO model version: V3,V4,V5,...")
    parser.add_argument("--input-size", type=int, default=640,
                        help="Image size for inference")
    parser.add_argument("--conf-thresh", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--nms_iou_thresh",type = float,default = 0.6,
                        help = "IoU threshold for Post Processing Non-Max Suppression")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device: cuda or cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    anchors = [[0.248,      0.7237237 ],
          [0.36144578, 0.53      ],
          [0.42,       0.9306667 ],
          [0.456,      0.6858006 ],
          [0.488,      0.8168168 ],
          [0.6636637,  0.274     ],
          [0.806,      0.648     ],
          [0.8605263,  0.8736842 ],
          [0.944,      0.5733333 ]]
    model = load_model(args.model_version,pretrained = args.pretrained,device = args.device,input_size = args.input_size,num_classes = 20, model_type = 'base',anchors = anchors)
    run_inference(model, args.device, args.source, args.input_size, args.conf_thresh,args.nms_iou_thresh)


if __name__ == "__main__":
    main()

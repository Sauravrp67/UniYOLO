#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import cv2
import torch
from utils import run_NMS,transform_xcycwh_to_x1y1x2y2,filter_confidence

# -------------------------------------------------------------------
# Import YOLO model families
# You can extend this with more (V4, V5, V6...) under model/
# -------------------------------------------------------------------
from model import YOLOv3
# from model.V4.yolo import YOLOv4
# from model.V5.yolo import YOLOv5

def load_model(version: str, weights: str, device: str):
    """
    Initialize YOLO model and load checkpoint.
    """
    if version.upper() == "V3":
        model = YOLOv3()
    # elif version.upper() == "V4":
    #     from model.V4.yolo import YOLOv4
    #     model = YOLOv4()
    # elif version.upper() == "V5":
    #     from model.V5.yolo import YOLOv5
    #     model = YOLOv5()
    else:
        raise ValueError(f"Unsupported model version: {version}")

    checkpoint = torch.load(weights, map_location=device)
    model.load_state_dict(checkpoint["model_state"] if "model_state" in checkpoint else checkpoint)
    model.to(device).eval()
    return model


def preprocess(image, img_size=640):
    """
    Basic resize/normalize. Adapt to your training pipeline.
    """
    img = cv2.resize(image, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0)  # [1,3,H,W]
    return img


def postprocess(preds, iou_threshold,conf_thresh=0.25):
    """
    Placeholder for postprocess (NMS etc.).
    For now, assume preds is already processed.
    """
    prediction = preds.cpu().numpy()
    prediction[:, 1:5] = transform_xcycwh_to_x1y1x2y2(boxes=prediction[:, 1:5], clip_max=1.0)
    prediction = filter_confidence(prediction=prediction, conf_threshold=conf_thresh)
    prediction = run_NMS(prediction=prediction, iou_threshold=iou_threshold)
    
    return prediction


def run_inference(model, device, source, img_size, conf_thresh):
    """
    Run inference on a source (image/video/webcam).
    """
    # Image
    if Path(source).suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"]:
        img = cv2.imread(source)
        t0 = time.time()
        inp = preprocess(img, img_size).to(device)
        preds = model(inp)
        preds = postprocess(preds, conf_thresh)
        fps = 1.0 / (time.time() - t0)
        print(f"FPS: {fps:.2f}")
        cv2.imshow("Prediction", img)
        cv2.waitKey(0)

    # Video/Webcam
    else:
        cap = cv2.VideoCapture(0 if source == "webcam" else source)
        assert cap.isOpened(), f"Failed to open {source}"

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t0 = time.time()
            inp = preprocess(frame, img_size).to(device)
            preds = model(inp)
            preds = postprocess(preds, conf_thresh)
            fps = 1.0 / (time.time() - t0)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Prediction", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
    cv2.destroyAllWindows()


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser("YOLO Inference")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to .pt checkpoint")
    parser.add_argument("--source", type=str, required=True,
                        help="Source: image.jpg | video.mp4 | webcam")
    parser.add_argument("--model-version", type=str, default="V3",
                        help="YOLO model version: V3,V4,V5,...")
    parser.add_argument("--img-size", type=int, default=640,
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
    model = load_model(args.model_version, args.weights, args.device)
    run_inference(model, args.device, args.source, args.img_size, args.conf_thresh)


if __name__ == "__main__":
    main()

import cv2
from inference_utils import post_process,denormalize,preprocess as preprocess_board,BoxDecoderNP,concat_heads_flatten_anchors,id2name,draw_dets

from pathlib import Path
import sys
import numpy as np
import time


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dataloader import BasicTransform,UnletterBox
from inference import preprocess

image = cv2.imread('./output_images/005688.jpg')
tp = BasicTransform(416)

print(image.dtype)

img_npy = np.load('./dump/input.npy')[0]
tensor0 = np.load('./dump/tensor0.npy')
tensor1 = np.load('./dump/tensor1.npy')
tensor2 = np.load('./dump/tensor2.npy')

print(tensor0.shape)
print(tensor1.shape)
print(tensor2.shape)


anchors = np.array([[0.248,      0.7237237 ],
        [0.36144578, 0.53      ],
        [0.42,       0.9306667 ],
        [0.456,      0.6858006 ],
        [0.488,      0.8168168 ],
        [0.6636637,  0.274     ],
        [0.806,      0.648     ],
        [0.8605263,  0.8736842 ],
        [0.944,      0.5733333 ]])

tensor0_decoded= BoxDecoderNP(tensor0,anchors[6:9]).decode_predictions()
tensor1_decoded= BoxDecoderNP(tensor1,anchors[3:6]).decode_predictions()
tensor2_decoded= BoxDecoderNP(tensor2,anchors[0:3]).decode_predictions()

print(tensor0_decoded.shape)
print(tensor1_decoded.shape)
print(tensor2_decoded.shape)

preds = np.concatenate([tensor0_decoded,tensor1_decoded,tensor2_decoded],axis = 1).squeeze(0)
print(preds.shape)

t0 = time.time()

image_out,boxes,labels,conf = post_process(
    img_npy,image_original=image,predictions = preds,tf = UnletterBox(new_shape = (416,416)),letterboxed = False,conf_thresh = 0.3,nms_iou_thresh=0.5
)
fps = 1.0 / (time.time() - t0)
names = id2name(labels) if labels is not None and len(labels) else []
vis = draw_dets(denormalize(image_out), boxes, labels, conf)
cv2.putText(vis, f"FPS: {fps:.2f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
print(vis.shape)
cv2.imshow("Prediction", vis)
cv2.waitKey(0)


# cv2.imshow('image',img_n]py)
# cv2.waitKey(0)
# cv2.imwrite('./output_images/cpu_preprocess.png',denormalize(img.numpy()))



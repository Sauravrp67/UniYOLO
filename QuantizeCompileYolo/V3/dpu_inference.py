import sys
import xir
import vart
import time
from typing import List
from ctypes import *
import random
import os

import cv2
import importlib
import numpy as np
from inference_utils import preprocess,draw_dets,id2name,BoxDecoderNP,denormalize,post_process
from transform import BasicTransform,AugmentTransform,UnletterBox
from pathlib import Path

def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."

    root_subgraph = graph.get_root_subgraph() # Retrieves the root subgraph of the input 'graph'
    assert (root_subgraph
            is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return [] # If it is a leaf, it means there are no child subgraphs, so the function returns an empty list 
    
    child_subgraphs = root_subgraph.toposort_child_subgraph() # Retrieves a list of child subgraphs of the 'root_subgraph' in topological order
    assert child_subgraphs is not None and len(child_subgraphs) > 0

    return [
        # List comprehension that filters the child_subgraphs list to include only those subgraphs that represent DPUs
        cs for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]

def run_model(dpu_runner,image):
    tensor_calculation_time = time.time()
    inputTensors = dpu_runner.get_input_tensors()
    outputTensors = dpu_runner.get_output_tensors()

    outputgrid0 = outputTensors[0].dims[1]
    outputanchors0 = outputTensors[0].dims[2]
    outputpreds0 = outputTensors[0].dims[3]

    outputgrid1 = outputTensors[1].dims[1]
    outputanchors1 = outputTensors[1].dims[2]
    outputpreds1 = outputTensors[1].dims[3]

    outputgrid2 = outputTensors[2].dims[1]
    outputanchors2 = outputTensors[2].dims[2]
    outputpreds2 = outputTensors[2].dims[3]

    outputSize_0 = (outputgrid0,outputanchors0,outputpreds0)
    outputSize_1 = (outputgrid1,outputanchors1,outputpreds1)
    outputSize_2 = (outputgrid2,outputanchors2,outputpreds2)

    runSize = 1
    shapeIn = (runSize,) + tuple([inputTensors[0].dims[i] for i in range(inputTensors[0].ndim)][1:])
    
    outputData = []
    inputData = []

    outputData.append(np.empty((runSize,*outputSize_0),dtype = np.float32,order='C'))
    outputData.append(np.empty((runSize,*outputSize_1),dtype = np.float32,order='C'))
    outputData.append(np.empty((runSize,*outputSize_2),dtype = np.float32,order='C'))

    inputData.append(np.empty((shapeIn),dtype = np.float32, order = "C"))
    tensor_calculation_end  = (time.time() - tensor_calculation_time) * 1000
    print(f"TensorShape Aquisition from DPU graph:{tensor_calculation_end:.2f}")
    """
    input buffer
    """
    filling_buffer_start = time.time()
    imageRun = inputData[0]
    imageRun[0,...] = image.reshape(inputTensors[0].dims[1],inputTensors[0].dims[2],inputTensors[0].dims[3])
    filling_buffer_end = (time.time() - filling_buffer_start) * 1000
    # .reshape(inputTensors[0].dims[1],inputTensors[0].dims[2],inputTensors[0].dims[3])
    print(f"Fill Buffer:{filling_buffer_end:.2f}")

    running_model = time.time()
    """Execute Async"""
    job_id = dpu_runner.execute_async(inputData,outputData)
    dpu_runner.wait(job_id)
    running_model_end = (time.time() - running_model) * 1000
    print(f"Running Model :{running_model_end:.2f}")

    return outputData,inputData[0]

# def run_model(dpu_runner, image):
#     """
#     image: float32 NHWC letterboxed+normalized frame from your preprocess()
#     Returns:
#       - output_list: list of 3 float32 NHWC (1, G, A, P) tensors (dequantized)
#       - input_nhwc:  float32 NHWC (1, H, W, C) copy of 'image' for downstream use
#     """
#     import numpy as np
#     import time

#     # ----- one-time setup (cached) -----
#     if not hasattr(run_model, "_ctx"):
#         in_t  = dpu_runner.get_input_tensors()[0]     # XIR Tensor
#         out_t = dpu_runner.get_output_tensors()       # list[XIR Tensor]

#         def _get_fixpos(t):
#             for k in ("fix_point", "quantize_pos", "fix_pos"):
#                 if hasattr(t, "has_attr") and t.has_attr(k):
#                     return t.get_attr(k)
#             raise RuntimeError(f"No fix-point attr on tensor '{getattr(t, 'name', '')}'")

#         in_fixpos   = _get_fixpos(in_t)
#         out_fixpos  = [_get_fixpos(t) for t in out_t]

#         in_shape    = (1, *in_t.dims[1:])           # e.g. (1, 416, 416, 3) NHWC
#         out_shapes  = [(1, *t.dims[1:]) for t in out_t]

#         # Preallocate INT8 buffers in the shapes the DPU wants.
#         input_bufs  = [np.empty(in_shape,  dtype=np.int8, order="C")]
#         output_bufs = [np.empty(s,         dtype=np.int8, order="C") for s in out_shapes]

#         # Save context for reuse
#         run_model._ctx = {
#             "runner": dpu_runner,
#             "in_shape": in_shape,
#             "out_shapes": out_shapes,
#             "input_bufs": input_bufs,
#             "output_bufs": output_bufs,
#             "in_scale": float(1 << int(in_fixpos)),
#             "out_scales": [float(1 << int(fp)) for fp in out_fixpos],
#         }

#     ctx = run_model._ctx

#     # ----- prepare input (quantize to INT8, NHWC) -----
#     # Make sure the incoming image is HWC float32 with expected spatial size.
#     HWC = ctx["in_shape"][1:]
#     if image.shape != HWC:
#         # Keep behavior identical to your old code (reshape if needed)
#         img_f32 = image.reshape(HWC).astype(np.float32, copy=False)
#     else:
#         img_f32 = image.astype(np.float32, copy=False)

#     # Quantize: symmetric INT8, zero-point = 0
#     # q = round(x * scale) clipped to [-128, 127]
#     q = np.rint(img_f32 * ctx["in_scale"]).clip(-128, 127).astype(np.int8)
#     # Single copy into the preallocated DPU input buffer
#     np.copyto(ctx["input_bufs"][0][0], q)

#     # ----- run DPU -----
#     jid = ctx["runner"].execute_async(ctx["input_bufs"], ctx["output_bufs"])
#     ctx["runner"].wait(jid)

#     # ----- dequantize outputs back to float32 (so downstream stays unchanged) -----
#     outs_f32 = [
#         buf.astype(np.float32) / scale
#         for buf, scale in zip(ctx["output_bufs"], ctx["out_scales"])
#     ]

#     # Your downstream expects the second return to be the float32 input batched
#     input_nhwc = img_f32.reshape(ctx["in_shape"])

#     return outs_f32, input_nhwc


def run_inference(dpu_runner,source,input_size,mode,conf_thresh,nms_iou_thresh,anchors):
    tf_pre = BasicTransform(input_size = input_size)
    tf_unlb = UnletterBox(new_shape = input_size)
    
    if mode == 'image':
    #------Image------
        if Path(source).suffix.lower() in [".jpg",".png",".jpeg",".bmp"]:

            img = cv2.imread(source)
            print(img.dtype)
            preprocess_start = time.time()
            preprocessed_image = preprocess(img,tf = tf_pre)
            preprocess_time = (time.time() - preprocess_start) * 1000

            model_time_start = time.time()
            output_list,input_image = run_model(dpu_runner = dpu_runner,image = preprocessed_image)
            model_time = (time.time() - model_time_start) * 1000
            
            tensor0_decoded= BoxDecoderNP(output_list[0],anchors[6:9]).decode_predictions()
            tensor1_decoded= BoxDecoderNP(output_list[1],anchors[3:6]).decode_predictions()
            tensor2_decoded= BoxDecoderNP(output_list[2],anchors[0:3]).decode_predictions()

            preds = np.concatenate([tensor0_decoded,tensor1_decoded,tensor2_decoded],axis = 1).squeeze(0)
            post_process_start = time.time()
            image_out,boxes,labels,conf = post_process(
                input_image[0],image_original=img,predictions = preds,tf = tf_unlb,letterboxed = True,conf_thresh = conf_thresh,nms_iou_thresh=nms_iou_thresh
                )
            post_process_end = (time.time() - post_process_start) * 1000
            
            fps = 1.0 / (time.time() - preprocess_start)
            names = id2name(labels) if labels is not None and len(labels) else []
            print(f"FPS: {fps:.2f} (Preprocess time: {preprocess_time:.2f}\nModel Inference time: {model_time:.2f}\nPost Process:{post_process_end:.2f})| labels: {', '.join(sorted(set(names)))}" if names else f"FPS: {fps:.2f} (Preprocess time: {preprocess_time:.2f}\nModel Inference time: {model_time:.2f}\nPost Process:{post_process_end:.2f})| labels: none")
            vis = draw_dets(denormalize(image_out), boxes, labels, conf)
            cv2.putText(vis, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Prediction",vis)
            cv2.waitKey(0)

    else:
        cap = cv2.VideoCapture(0 if mode == 0 else source)
        assert cap.isOpened(), f"Failed to open{source}"
        while True:
            ret,img = cap.read()
            if not ret:
                break
            preprocess_start = time.time()
            preprocessed_image = preprocess(img,tf = tf_pre)
            preprocess_time = time.time() - preprocess_start

            model_time_start = time.time()
            output_list,input_image = run_model(dpu_runner = dpu_runner,image = preprocessed_image)
            model_time = time.time() - model_time_start

            tensor0_decoded= BoxDecoderNP(output_list[0],anchors[6:9]).decode_predictions()
            tensor1_decoded= BoxDecoderNP(output_list[1],anchors[3:6]).decode_predictions()
            tensor2_decoded= BoxDecoderNP(output_list[2],anchors[0:3]).decode_predictions()

            preds = np.concatenate([tensor0_decoded,tensor1_decoded,tensor2_decoded],axis = 1).squeeze(0)
            post_process_start = time.time()
            image_out,boxes,labels,conf = post_process(
                input_image[0],image_original=img,predictions = preds,tf = tf_unlb,letterboxed = True,conf_thresh = conf_thresh,nms_iou_thresh=nms_iou_thresh
                )
            post_process_end = time.time() - post_process_start
            
            fps = 1.0 / (time.time() - model_time_start)
            names = id2name(labels) if labels is not None and len(labels) else []
            print(f"FPS: {fps:.2f} (Preprocess time: {preprocess_time:.2f}\nModel Inference time: {model_time:.2f}\nPost Process:{post_process_end:.2f})| labels: {', '.join(sorted(set(names)))}" if names else f"FPS: {fps:.2f} (Preprocess time: {preprocess_time:.2f}\nModel Inference time: {model_time:.2f}\nPost Process:{post_process_end:.2f})| labels: none")
            vis = draw_dets(denormalize(image_out), boxes, labels, conf)
            cv2.putText(vis, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Prediction",vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

# class DpuRunnerCtx:
#     def __init__(self, runner):
#         self.runner = runner
#         self.in_t  = runner.get_input_tensors()[0]
#         self.out_t = runner.get_output_tensors()

#         # Shapes are NHWC on KV260 (e.g., [1, 416, 416, 3])
#         self.in_shape  = (1, *self.in_t.dims[1:])
#         self.out_shapes = [(1, *t.dims[1:]) for t in self.out_t]

#         # Preallocate INT8 buffers (what the DPU actually uses)
#         self.inputs  = [np.empty(self.in_shape, dtype=np.int8,  order='C')]
#         self.outputs = [np.empty(s,          dtype=np.int8,  order='C') for s in self.out_shapes]

#         # Quant scales from tensor attrs (Vitis uses symmetric, power-of-two scale)
#         self.in_fixpos  = self._get_fixpos(self.in_t)
#         self.out_fixpos = [self._get_fixpos(t) for t in self.out_t]
#         self.in_scale   = 1 << self.in_fixpos
#         self.out_scales = [1 << fp for fp in self.out_fixpos]

#     @staticmethod
#     def _get_fixpos(t):
#         if t.has_attr("fix_point"):
#             return t.get_attr("fix_point")
#         if t.has_attr("quantize_pos"):
#             return t.get_attr("quantize_pos")
#         raise RuntimeError("Tensor lacks fix_point/quantize_pos.")

#     def infer(self, img_f32_nhwc):
#         """
#         img_f32_nhwc: float32 NHWC after your mean/std normalization (or 0..1),
#         quantized here to int8 using in_scale.
#         Returns list of dequantized float32 outputs in NHWC.
#         """
#         # Quantize to int8 (symmetric, zero-point=0)
#         q = np.rint(img_f32_nhwc * self.in_scale).clip(-128, 127).astype(np.int8)
#         np.copyto(self.inputs[0][0], q)  # one copy into prealloc buffer

#         t0 = time.time()
#         jid = self.runner.execute_async(self.inputs, self.outputs)
#         self.runner.wait(jid)
#         t1 = time.time()

#         # Dequantize outputs (vectorized)
#         outs = [o.astype(np.float32) / s for o, s in zip(self.outputs, self.out_scales)]
#         return outs, (t1 - t0) * 1000.0  # ms

    
        
# python3 dpu_inference.py <xmodel_file> <image/video/webcam(1,2,0)> <image_path/video_path> <Confidence Threshold> <NMS IOU>"
def main(argv):
    os.makedirs("./dump",exist_ok = True)
    mode = argv[2]
    source_path = argv[3]
    conf_thresh = argv[4]
    nms_iou_thresh = argv[5]
    print(type(mode))
    anchors = np.array([[0.248,      0.7237237 ],
        [0.36144578, 0.53      ],
        [0.42,       0.9306667 ],
        [0.456,      0.6858006 ],
        [0.488,      0.8168168 ],
        [0.6636637,  0.274     ],
        [0.806,      0.648     ],
        [0.8605263,  0.8736842 ],
        [0.944,      0.5733333 ]])
    g = xir.Graph.deserialize(argv[1])
    subgraphs = get_child_subgraph_dpu(g)

    #"""Creates DPU runner,associated with the DPU subgraph"""
    dpu_runners = vart.Runner.create_runner(subgraphs[0],"run")
    run_inference(dpu_runner = dpu_runners,source = source_path,mode = mode,input_size = 416,conf_thresh = 0.3,nms_iou_thresh = 0.5,anchors=anchors)

    # print(input.shape)
    # print(output_list[0].shape)
    # print(output_list[1].shape)
    # print(output_list[2].shape)
    
    # np.save("./dump/input.npy",input)
    # np.save("./dump/tensor0.npy",output_list[0])
    # np.save("./dump/tensor1.npy",output_list[1])
    # np.save("./dump/tensor2.npy",output_list[2])

    # print("Saurav")
    # print(subgraphs)
    # print(dpu_runners)
    
    # input_tensors = dpu_runners.get_input_tensors()
    # output_tensor = dpu_runners.get_output_tensors()

    # print(input_tensors[0].dims)
    # print(output_tensor[0].dims)
    # print(output_tensor[1].dims)
    # print(output_tensor[2].dims)



if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("usage : python3 dpu_inference.py <xmodel_file> <image/video> <image_path/video_path> <Confidence Threshold> <NMS IOU>")
    else:
        main(sys.argv)


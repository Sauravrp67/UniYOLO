import sys
import xir
import vart
import time
from typing import List
from ctypes import *
import random
import os

from pathlib import Path
import cv2
import numpy as np

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

def run_model(dpu_runner, image):
    t0 = time.perf_counter()
    inputTensors = dpu_runner.get_input_tensors()
    t1 = time.perf_counter()
    outputTensors = dpu_runner.get_output_tensors()
    t2 = time.perf_counter()

    outputShape = outputTensors[0].dims[1]
    t3 = time.perf_counter()
    runSize = 1
    shapeIn = (runSize,) + tuple([inputTensors[0].dims[i] for i in range(inputTensors[0].ndim)][1:])
    t4 = time.perf_counter()

    outputData = []
    inputData = []
    t5 = time.perf_counter()

    outputData.append(np.empty((runSize, outputShape), dtype=np.float32, order='C'))
    t6 = time.perf_counter()
    inputData.append(np.empty((shapeIn), dtype=np.float32, order="C"))
    t7 = time.perf_counter()

    imageRun = inputData[0]
    imageRun[0, ...] = image.reshape(inputTensors[0].dims[1],
                                     inputTensors[0].dims[2],
                                     inputTensors[0].dims[3])
    t8 = time.perf_counter()

    job_id = dpu_runner.execute_async(inputData, outputData)
    t9 = time.perf_counter()
    dpu_runner.wait(job_id)
    t10 = time.perf_counter()

    # Report timings in ms
    print(f"get_input_tensors: {(t1 - t0) * 1000:.3f} ms")
    print(f"get_output_tensors: {(t2 - t1) * 1000:.3f} ms")
    print(f"outputShape assign: {(t3 - t2) * 1000:.3f} ms")
    print(f"shapeIn calculation: {(t4 - t3) * 1000:.3f} ms")
    print(f"lists init: {(t5 - t4) * 1000:.3f} ms")
    print(f"alloc outputData: {(t6 - t5) * 1000:.3f} ms")
    print(f"alloc inputData: {(t7 - t6) * 1000:.3f} ms")
    print(f"buffer filling: {(t8 - t7) * 1000:.3f} ms")
    print(f"execute_async enqueue: {(t9 - t8) * 1000:.3f} ms")
    print(f"DPU wait: {(t10 - t9) * 1000:.3f} ms")
    print(f"TOTAL run_model: {(t10 - t0) * 1000:.3f} ms")

    return outputData, inputData[0]

MEAN = 0.485, 0.456, 0.406 # RGB
STD = 0.229, 0.224, 0.225 

class Normalize:
    def __init__(self,mean,std):
        self.mean = np.array(mean,dtype = np.float32)
        self.std = np.array(std,dtype = np.float32)
    
    def __call__(self,image,boxes = None,labels = None):
        image /= 255
        image -= self.mean
        image /= self.std
        return image,boxes,labels

def softmax(logits, axis=-1):
    x = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def load_labels(path):
    with open(path, "r") as f:
        return [line.strip() for line in f]

def main(argv):
    os.makedirs("./dump",exist_ok = True)
    mode = argv[2]
    source_path = argv[3]

    g = xir.Graph.deserialize(argv[1])
    subgraphs = get_child_subgraph_dpu(g)

    runner_create_t0 = time.perf_counter()
    dpu_runners = vart.Runner.create_runner(subgraphs[0], "run")
    runner_create_ms = (time.perf_counter() - runner_create_t0) * 1000.0

    image = cv2.imread(source_path)
    t0 = time.perf_counter()
    image = cv2.resize(image,(224,224),interpolation = cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image,_,_ = Normalize(mean = MEAN,std=STD)(image = image,boxes = None,labels=None)
    preprocess = (time.perf_counter()  - t0) * 1000

    infer_t0 = time.perf_counter()
    output, input_buf = run_model(dpu_runners, image)
    infer_ms = (time.perf_counter() - infer_t0) * 1000.0

    post_t0 = time.perf_counter()
    cls = softmax(output[0])                # shape (1, 10)
    pred_idx = np.argmax(cls, axis=-1).item()  # <- avoid deprecation warning
    post_ms = (time.perf_counter() - post_t0) * 1000.0

    labels = load_labels("labels.txt")
    pred_name = labels[pred_idx] if pred_idx < len(labels) else str(pred_idx)

    disp_img = cv2.imread(source_path)
    disp_img = cv2.resize(disp_img, (224, 224))

    conf = float(np.max(cls))
    total_elapsed_s = (time.perf_counter() - t0)   # t0 was pre-processing start
    fps = 1.0 / total_elapsed_s if total_elapsed_s > 0 else 0.0

    cv2.putText(disp_img, f"{pred_name}: {conf:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(disp_img, f"FPS: {fps:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    print(f"Runner create: {runner_create_ms:.3f} ms")
    print(f"Model inference (run_model): {infer_ms:.3f} ms")
    print(f"Pre-process: {preprocess:.3f} ms")
    print(f"Post-process: {post_ms:.3f} ms")

    print(f"FPS:{fps}")
    cv2.imshow("Prediction", disp_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage : python3 dpu_inference.py <xmodel_file> <image/video> <image_path/video_path>")
    else:
        main(sys.argv)
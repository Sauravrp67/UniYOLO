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

    # inputData.append(np.empty((shapeIn),dtype = np.float32, order = "C"))
    # tensor_calculation_end  = (time.time() - tensor_calculation_time) * 1000
    # print(f"TensorShape Aquisition from DPU graph:{tensor_calculation_end:.2f}")
    # """
    # input buffer
    # """
    # filling_buffer_start = time.time()
    # imageRun = inputData[0]
    # imageRun[0,...] = image.reshape(inputTensors[0].dims[1],inputTensors[0].dims[2],inputTensors[0].dims[3])
    # filling_buffer_end = (time.time() - filling_buffer_start) * 1000
    # # .reshape(inputTensors[0].dims[1],inputTensors[0].dims[2],inputTensors[0].dims[3])
    # print(f"Fill Buffer:{filling_buffer_end:.2f}")

    # running_model = time.time()
    # """Execute Async"""
    # job_id = dpu_runner.execute_async(inputData,outputData)
    # dpu_runner.wait(job_id)
    # running_model_end = (time.time() - running_model) * 1000
    # print(f"Running Model :{running_model_end:.2f}")

    return outputData,inputData[0]

MEAN = 0.485, 0.456, 0.406 # RGB
STD = 0.229, 0.224, 0.225 

class Normalize:
    def __init__(self,mean,std):
        self.mean = np.array(mean,dtype = np.float32)
        self.std = np.array(std,dtype = np.float32)
    
    def __call__(self,image,boxes = None,labels = None):
        image /= 255.0
        image -= self.mean
        image /= self.std
        return image,boxes,labels

def main(argv):
    os.makedirs("./dump",exist_ok = True)
    mode = argv[2]
    source_path = argv[3]
    # conf_thresh = argv[4]
    # nms_iou_thresh = argv[5]
    
    g = xir.Graph.deserialize(argv[1])
    subgraphs = get_child_subgraph_dpu(g)
    image = cv2.imread(source_path)
    image = cv2.resize(image,(224,224),interpolation = cv2.INTER_LINEAR)
    image,_,_ = Normalize(mean = MEAN,std=STD)(image = image,boxes = None,labels=None)


    print(image.shape)
    #"""Creates DPU runner,associated with the DPU subgraph"""
    dpu_runners = vart.Runner.create_runner(subgraphs[0],"run")
    output,input = run_model(dpu_runners,image)
    
    print(output)
    print(input.shape)
    # run_inference(dpu_runner = dpu_runners,source = source_path,mode = mode,input_size = 416,conf_thresh = 0.3,nms_iou_thresh = 0.5,anchors=anchors)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage : python3 dpu_inference.py <xmodel_file> <image/video> <image_path/video_path>")
    else:
        main(sys.argv)
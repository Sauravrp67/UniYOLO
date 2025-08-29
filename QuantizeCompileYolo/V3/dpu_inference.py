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
from inference_utils import preprocess,draw_dets
from transform import BasicTransform,AugmentTransform,UnletterBox

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
    
    "Buffers"

    outputData = []
    inputData = []

    outputData.append(np.empty((runSize,*outputSize_0),dtype = np.float32,order='C'))
    outputData.append(np.empty((runSize,*outputSize_1),dtype = np.float32,order='C'))
    outputData.append(np.empty((runSize,*outputSize_2),dtype = np.float32,order='C'))

    inputData.append(np.empty((shapeIn),dtype = np.float32, order = "C"))

    """
    input buffer
    """

    imageRun = inputData[0]
    imageRun[0,...] = image.reshape(inputTensors[0].dims[1],inputTensors[0].dims[2],inputTensors[0].dims[3])

    """Execute Async"""
    job_id = dpu_runner.execute_async(inputData,outputData)
    dpu_runner.wait(job_id)

    return outputData,inputData[0]

def main(argv):
    os.makedirs("dump",exist_ok = True)
    g = xir.Graph.deserialize(argv[1])
    subgraphs = get_child_subgraph_dpu(g)
    image_path = argv[2]

    img_numpy = cv2.imread(image_path)
    
    tf_pre = BasicTransform(input_size = 416)
    tf_unlb = UnletterBox(new_shape = 416)

    preprocessed_image = preprocess(img_numpy,tf = tf_pre)

    cv2.imwrite('./transformed_image.png',preprocessed_image)

    assert len(subgraphs) == 1

    #"""Creates DPU runner,associated with the DPU subgraph"""
    dpu_runners = vart.Runner.create_runner(subgraphs[0],"run")
    output_list,input =run_model(dpu_runner=dpu_runners,image = preprocessed_image)
    print(input.shape)
    print(output_list[0].shape)
    print(output_list[1].shape)
    print(output_list[2].shape)
    
    np.save("dump/input.npy",input)
    np.save("dump/tensor0.npy",output_list[0])
    np.save("dump/tensor1.npy",output_list[1])
    np.save("dump/tensor2.npy",output_list[2])

    print("Saurav")
    print(subgraphs)
    print(dpu_runners)
    
    input_tensors = dpu_runners.get_input_tensors()
    output_tensor = dpu_runners.get_output_tensors()

    print(input_tensors[0].dims)
    print(output_tensor[0].dims)
    print(output_tensor[1].dims)
    print(output_tensor[2].dims)



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage : python3 dpu_inference.py <xmodel_file> <image_path>")
    else:
        main(sys.argv)


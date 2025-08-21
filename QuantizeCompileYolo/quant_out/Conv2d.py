# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class Conv2d(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(Conv2d, self).__init__()
        self.module_0 = py_nndct.nn.Input() #Conv2d::input_0(Conv2d::nndct_input_0)
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Conv2d::Conv2d/ret(Conv2d::nndct_conv2d_1)

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        return output_module_0

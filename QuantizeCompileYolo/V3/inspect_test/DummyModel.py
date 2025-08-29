# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class DummyModel(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.module_0 = py_nndct.nn.Input() #DummyModel::input_0(DummyModel::nndct_input_0)
        self.module_1 = py_nndct.nn.Module('nndct_reshape') #DummyModel::DummyModel/Reshape[reshape]/ret.3(DummyModel::nndct_reshape_1)
        self.module_2 = py_nndct.nn.Module('nndct_contiguous') #DummyModel::DummyModel/Reshape[reshape]/54(DummyModel::nndct_contiguous_2)
        self.module_3 = py_nndct.nn.Add() #DummyModel::DummyModel/ret(DummyModel::nndct_elemwise_add_3)
        self.grid_x4d = torch.nn.parameter.Parameter(torch.Tensor(1, 13, 13, 3))

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(input=output_module_0, shape=[1,13,13,3])
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(input=output_module_0, other=self.grid_x4d, alpha=1)
        return output_module_0

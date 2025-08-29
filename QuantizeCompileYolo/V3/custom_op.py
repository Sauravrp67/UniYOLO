import torch.nn as nn
import torch
from pytorch_nndct.apis import Inspector
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from Paper_Implementation.ObjectDetection.UniYOLO.QuantizeCompileYolo.V3.quantize_utils import set_grid

# class elementwise_mul(nn.modules):
#     def __init__(self,const,tensor):
#         super(elementwise_mul,self).__init__()
#         self.shape = tensor.shape
#         const_tensor = torch.fill(self.shape,const)
#         print(const_tensor)

anchors = [[0.248,      0.7237237 ],
            [0.36144578, 0.53      ],
            [0.42,       0.9306667 ],
            [0.456,      0.6858006 ],
            [0.488,      0.8168168 ],
            [0.6636637,  0.274     ],
            [0.806,      0.648     ],
            [0.8605263,  0.8736842 ],
            [0.944,      0.5733333 ]]

# anchors = torch.tensor(anchors)

# import torch
# import torch.nn as nn

# class Reshape(nn.Module):
#     def forward(self, x):
#         return x.view(1, 13, 13, 3).contiguous()

# class DummyModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.stride = 32
#         self.reshape = Reshape()

#         # Register buffers once
#         self.register_buffer("grid_x4d", torch.empty(0))           # (1,13,13,3)
#         self.register_buffer("invS", torch.ones(1, dtype=torch.float32))
#         self._set_grid_xy(416)

#     @torch.no_grad()
#     def _set_grid_xy(self, input_size):
#         S = input_size // self.stride                              # 13
#         A = 3
#         gy, gx = torch.meshgrid(torch.arange(S), torch.arange(S), indexing="ij")
#         gx4 = gx.to(torch.float32).view(1, S, S, 1).expand(1, S, S, A).contiguous()
#         self.grid_x4d.resize_(gx4.shape).copy_(gx4)                # mutate buffer, no reassignment
#         self.invS.fill_(1.0 / float(S))

#     def forward(self, x1):
#         x1 = self.reshape(x1)                                      # (1,13,13,3)
#         # Now add is input (feature map) + DPU constant buffer → DPU
#         return x1 @ self.grid_x4d                                  # or (x1 + self.grid_x4d) * self.invS


tensorA = torch.randn((1,169,3))
tensorB = torch.randn((1,169,3))
tensorC = torch.randn((1,169,3))
tensorD = torch.randn((1,169,3))

final_tensor = torch.cat((tensorA.unsqueeze(-1),tensorB.unsqueeze(-1),tensorC.unsqueeze(-1),tensorD.unsqueeze(-1)),dim = -1)
print(final_tensor[:,50,:])
print(tensorA[:,50,:])
print(tensorB[:,50,:])
print(tensorC[:,50,:])
print(tensorD[:,50,:])



# dummy_tensor = torch.fill(torch.zeros((1,169,3,1)),value = 5)

# print(dummy_tensor.expand(1,169,3,20))

# model = DummyModel().eval()
# target = "0x101000056010407"
# inspector = Inspector(target)

# dummy_input1 = torch.randn(1,169,3)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# inspector.inspect(model,(dummy_input1,),device = device,output_dir = "./inspect_test",verbose_level = 2,image_format = "png")


        


    


from torch import Tensor,nn
import torch
import numpy as np
from pytorch_nndct.expanding.structured import ExpandingRunner


m = nn.Sequential(
    nn.Conv2d(3, 58, 3, padding=1),
    nn.Conv2d(58, 29, 1),
)
runner = ExpandingRunner(m, torch.randn(1,3,224,224))
m_exp, meta = runner.expand(64)   # align to multiple of 64
print(m)       # 58 -> 29
print(m_exp)  
print(meta) # will show 64-aligned internals
# Verify functional equivalence
x = torch.randn(1,3,224,224)

unexpanded_result = m.eval()(x)
expanded_result = m_exp.eval()(x)

print(torch.allclose(unexpanded_result,expanded_result[:,:unexpanded_result.shape[1]]))
print(expanded_result[:,unexpanded_result.shape[1]].abs().max() == 0)

# print(torch.allclose(m.eval()(x), m_exp.eval()(x), atol=1e-6))
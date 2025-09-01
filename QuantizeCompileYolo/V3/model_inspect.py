import os
import sys
import argparse


from pathlib import Path
import sys
import torch
from pytorch_nndct.apis import Inspector

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from model import YOLOv3,YOLOv3_DPU

anchors = [[0.248,      0.7237237 ],
    [0.36144578, 0.53      ],
    [0.42,       0.9306667 ],
    [0.456,      0.6858006 ],
    [0.488,      0.8168168 ],
    [0.6636637,  0.274     ],
    [0.806,      0.648     ],
    [0.8605263,  0.8736842 ],
    [0.944,      0.5733333 ]]

model = YOLOv3(input_size = 416,num_classes = 20,anchors = anchors,model_type = "base")

dummy_input = torch.randn(1,3,416,416)
target = "0x101000056010407"

inspector = Inspector(target)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inspector.inspect(model,(dummy_input,),device = device,output_dir = "./inspect",verbose_level = 2,image_format = "png")




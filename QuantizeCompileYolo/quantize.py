from pathlib import Path
import sys
import torch
from pytorch_nndct.apis import torch_quantizer
from pytorch_nndct.apis import Inspector
import cv2

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from model import YOLOv3

print(YOLOv3())

model = torch.nn.Conv2d(3,16,3).eval()

dummy = torch.randn(1,3,224,224)


target = "DPUCZDX8G_ISA1_B4096"

inspector = Inspector(target)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inspector.inspect(model, (dummy,), device=device, output_dir="./inspect", image_format="png") 


# quantizer = torch_quantizer("calib",model,(dummy,),output_dir = "./quant_out")

# qmodel = quantizer.quant_model

# print("Quantized Model Ready:",type(qmodel))

# print(torch_quantizer)
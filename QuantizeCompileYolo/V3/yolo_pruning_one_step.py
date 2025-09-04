from pytorch_nndct import get_pruning_runner
from pytorch_nndct import OFAPruner
import torch
from quantize_utils import load_model,evaluate,get_dataloader
from tqdm import tqdm
import argparse

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
   sys.path.append(str(ROOT))

from utils import YOLOv3Loss

parser = argparse.ArgumentParser("YOLO Pruning")
parser.add_argument("--data",
                  type = str,
                  required = True
                  )
parser.add_argument("--subset",
                  type = int,
                  required = False,
                  default = 100)

parser.add_argument("--model-path", type=str, required=True,
                        help="Path to .pt checkpoint")

parser.add_argument("--model-version", type=str, default="V3",
                  help="YOLO model version: V3,V4,V5,...")

parser.add_argument("--input-size", type=int, default=640,
            help="Image size for inference")

parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                  help="Device: cuda or cpu")

parser.add_argument("--mode",type = str,help = "type of model to run")

args = parser.parse_args()


anchors = [
   [0.248,      0.7237237 ],
   [0.36144578, 0.53      ],
   [0.42,       0.9306667 ],
   [0.456,      0.6858006 ],
   [0.488,      0.8168168 ],
   [0.6636637,  0.274     ],
   [0.806,      0.648     ],
   [0.8605263,  0.8736842 ],
   [0.944,      0.5733333 ]
      ]
model = load_model(args.mode,model_path=args.model_path,device = args.device,input_size = args.input_size,num_classes= 20,model_type="base",anchors = anchors)

loader = get_dataloader(voc_path = args.data,batch_size = 8,subset_length = args.subset,train = False,input_size=args.input_size)

input_signature = torch.randn([1,3,416,416],dtype = torch.float32)
runner = get_pruning_runner(model,input_signature,'one_step')

criterion = YOLOv3Loss(input_size=416,num_classes = 20,anchors = model.anchors)


def calibration_fn(model, dataloader, number_forward=100):
  model.train()
  with torch.no_grad():
    for index, (_,images, target,_) in enumerate(dataloader):
      images = images.to('cpu')
      model(images)
      if index > number_forward:
        break


runner.search(gpus=[], calibration_fn=calibration_fn, calib_args=(loader,), eval_fn=evaluate, eval_args=(loader,criterion,), num_subnet=10, removal_ratio=0.7)

model = runner.prune(removal_ratio = 0.2)
print(model)
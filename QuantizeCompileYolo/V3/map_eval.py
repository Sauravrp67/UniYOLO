import sys
from pathlib import Path
import argparse
import torch
from pytorch_nndct.apis import torch_quantizer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import Evaluator,generate_random_color
from dataloader import Dataset,BasicTransform
from quantize_utils import load_model,get_dataloader,validate
from model import BoxDecoder


parser = argparse.ArgumentParser()

parser.add_argument(
    '--model-path',
    default = None,
    help = 'path to model to evaluate'
)

parser.add_argument(
    '--img-size',type = int,default = 416,help = "Model input size"
)

parser.add_argument(
    '--conf-thres',
    type = float,
    default = 0.25,
    help = "confidence threshold for a prediction to be consider true positive"
)

parser.add_argument(
    '--nms-thres',
    type = float,
    default = 0.6,
    help = "iou threshold for nms"
)

parser.add_argument(
    '--exp-path',
    type = str,
    default = './mAP_results',

)

parser.add_argument(
    '--data',
    type = str,
    default = None,
    help = 'path to yaml file'

)

args,_ = parser.parse_known_args()
# Get stored the require value for validate in args.

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


model_dpu = load_model(mode = 'dpu_quantized',device = 'cpu',input_size = 416,num_classes = 20,model_type = "base",anchors = anchors,model_path = args.model_path)
# model = torch.jit.load('/workspace/QuantizeCompileYolo/V3/quantize_result/YOLOv3_DPU_int.pt',map_location='cpu')
val_loader = get_dataloader(args.data,batch_size = 8,subset_length = 200,train = False)
# print(model_dpu)

# _,image,_,_ = next(iter(val_loader))

# preds = model_dpu(image)

# decoded_52 = BoxDecoder(preds[0],torch.tensor(anchors[0:3])).decode_predictions()
# decoded_26 = BoxDecoder(preds[1],torch.tensor(anchors[3:6])).decode_predictions()
# decoded_13 = BoxDecoder(preds[2],torch.tensor(anchors[6:9])).decode_predictions()
# preds = torch.cat((decoded_52,decoded_26,decoded_13),dim = 1)

# print(preds.shape)

subset_class = val_loader.dataset
base_dataset_class = val_loader.dataset.dataset
idx = list(subset_class.indices)
print(idx)

base_dataset_class.generate_mAP_source(save_dir = Path("./data/eval_src"),mAP_filename = "subset.json",indices = idx)

args.mAP_filepath = Path(base_dataset_class.mAP_filepath) 
args.exp_path = Path(args.exp_path)

evaluator = Evaluator(args.mAP_filepath)
val_loader = tqdm(val_loader, desc="[VAL]", ncols=115, leave=False)

mAP_dict,eval_text = validate(args,anchors = anchors, dataloader = val_loader,model = model_dpu,evaluator = evaluator,save_result = True,dpu = True,save_filename="quantized_dpu_results.txt")

print(mAP_dict)
print(eval_text)


# val_loader = get_dataloader(batch_size = 4,subset_length = 8,train = False)








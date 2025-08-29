import sys
import argparse
import time
from pytorch_nndct.apis import torch_quantizer
import torch

from tqdm import tqdm

from quantize_utils import get_dataloader,load_model,evaluate

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from model import YOLOv3_DPU,YOLOv3
from dataloader import Dataset,BasicTransform,AugmentTransform
from torch.utils.data import DataLoader,Subset
from utils import YOLOv3Loss
from val import validate

parser = argparse.ArgumentParser()

parser.add_argument(
    '--config-file',
    default=None,
    help='quantization configuration file')
parser.add_argument(
    '--subset-len',
    default=None,
    type=int,
    help='subset_len to evaluate model, using the whole validation dataset if it is not set')
parser.add_argument(
    '--batch-size',
    default=8,
    type=int,
    help='input data batch size to evaluate model')

parser.add_argument('--quant-mode', 
    default='calib', 
    choices=['float', 'calib', 'test'], 
    help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')

parser.add_argument('--fast-finetune', 
    dest='fast_finetune',
    action='store_true',
    help='fast finetune model before calibration')


parser.add_argument('--deploy', 
    dest='deploy',
    action='store_true',
    help='export xmodel for deployment')


parser.add_argument('--target', 
    dest='target',
    nargs="?",
    const="",
    help='specify target device')


parser.add_argument(
    "--data",
    type = str,
    default = None,
    help = 'path to YAML file'
    )

parser.add_argument("--device",default = "cpu",help = "set device cpu or gpu")

parser.add_argument("--model-name",default = "V3",help = "Model to quantize")

parser.add_argument("--model-path",
                    type = str,
                    default = None,
                    help = "path to model's .pt file")

args, _ = parser.parse_known_args()

def quantization(title='optimize',
                 model_name='', 
                 file_path=''): 
    quant_mode = args.quant_mode
    deploy = args.deploy
    batch_size = args.batch_size
    config_file = args.config_file
    finetune = args.fast_finetune
    subset_length = args.subset_len
    data = args.data
    device = torch.device(args.device)

    if quant_mode != 'test' and deploy:
        deploy = False
        print(r'Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!')
    if deploy and (batch_size != 1):
        print(r'Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!')
        batch_size = 1

    #Load the model
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

    model = load_model(mode = "dpu",device=args.device,input_size = 416,num_classes = 20,model_type = "base",anchors = anchors,model_path = args.model_path)

    input_sig = torch.randn([batch_size,3,416,416]).to(device)

    if quant_mode == 'float':
        quant_model = model
    else:
        quantizer = torch_quantizer(
            quant_mode, model, (input_sig,), device=device, quant_config_file=config_file)

        quant_model = quantizer.quant_model.eval()
    
    if quant_mode == 'calib':
        calib_loader = get_dataloader(voc_path = data,batch_size = batch_size,subset_length = subset_length,train = False)
        quant_model.eval()
        with torch.no_grad():
            for _,imgs,_,_ in tqdm(calib_loader,desc = 'Calibrating'):
                imgs = imgs.to(device,non_blocking=True)
                _ = quant_model(imgs)
        
        criterion = YOLOv3Loss(input_size=416,num_classes = 20,anchors = model.anchors)
   
        if finetune == True:
            ft_loader = get_dataloader(
                voc_path=data,
                batch_size = batch_size,
                subset_length = subset_length,
                train = False
            )

            quantizer.fast_finetune(evaluate,(quant_model,ft_loader,criterion))
        quantizer.export_quant_config()
    
    elif quant_mode == 'test':
        if quant_mode == finetune:
            quantizer.load_ft_param()   # only exists if calib+fast_finetune ran earlier
            print("Loaded fast-finetune params.")
        with torch.no_grad():
            _ = quant_model(input_sig)

        if deploy:
            quantizer.export_torch_script()
            quantizer.export_onnx_model()
            quantizer.export_xmodel(deploy_check=True, dynamic_batch=True)


if __name__ == '__main__':    
    model_name = args.model_name

    feature_test = ' float model evaluation'
    if args.quant_mode != 'float':
        feature_test = ' quantization'
        # force to merge BN with CONV for better quantization accuracy
        args.optimize = 1
        feature_test += ' with optimization'
    else:
        feature_test = ' float model evaluation'
    title = model_name + feature_test
    print("-------- Start {} test ".format(model_name))
    quantization(
        title=title,
        model_name=model_name,
        file_path='')
    print("-------- End of {} test ".format(model_name))










from torchvision.models.resnet import resnet18
from pytorch_nndct import get_pruning_runner
from pytorch_nndct import OFAPruner
import torch
from torch.utils.data import Subset,DataLoader
import cv2
from tqdm import tqdm

import sys
from pathlib import Path
from quantize_utils import load_model,evaluate,get_dataloader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
   sys.path.append(str(ROOT))

from dataloader import Dataset,AugmentTransform
from quantize_utils import to_image


model = resnet18(pretrained=True).to('cpu')
input_signature = torch.randn([1,3,224,224],dtype = torch.float32)
runner = get_pruning_runner(model,input_signature,'one_step')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} (avg: {avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # get top-k indices
        _, pred = output.topk(maxk, 1, True, True)   # [B, maxk]
        pred = pred.t()                              # [maxk, B]
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def eval_fn(model, dataloader):
  top1 = AverageMeter('Acc@1', ':6.2f')
  model.eval().to('cpu')
  with torch.no_grad():
    for i, (_,images, targets,_) in enumerate(dataloader):
        images = images.to('cpu')
        targets = torch.randint(low=0, high=100, size=(4,), device='cpu')
        outputs = model(images)
        acc1, _ = accuracy(outputs, targets, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
    return top1.avg


def calibration_fn(model, dataloader, number_forward=100):
  model.train()
  with torch.no_grad():
    for index, (_,images, target,_) in enumerate(dataloader):
      images = images.to('cpu')
      model(images)
      if index > number_forward:
        break


train_loader = get_dataloader('/workspace/data/voc_vitis.yaml',batch_size = 8)

best_idx = runner.search(gpus=[], calibration_fn=calibration_fn, calib_args=(train_loader,), eval_fn=eval_fn, eval_args=(train_loader,), num_subnet=10, removal_ratio=0.7)

# model = runner.prune(removal_ratio=0.7, index=None)

# runner.ana(eval_fn, args=(train_loader,),gpus = None)

model = runner.prune(removal_ratio=0.2,index = best_idx)




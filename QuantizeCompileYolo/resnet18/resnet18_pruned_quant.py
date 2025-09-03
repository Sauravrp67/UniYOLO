import argparse
import os
import time
from pathlib import Path
from typing import Tuple, Optional


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms,models
from tqdm import tqdm
import torch_pruning as tp
from pytorch_nndct import torch_quantizer

class AverageMeter:
    def __init__(self):
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
        self.avg = self.sum / max(1, self.count)

@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)) -> Tuple[float, float]:
    """Compute top-k accuracy (supports fewer than k classes)."""
    maxk = max(min(k, output.size(1)) for k in topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        k = min(k, output.size(1))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append((correct_k.mul_(100.0 / batch_size)).item())
    return tuple(res)

def cifar10_loaders(data_dir: str, batch_size: int = 128, workers: int = 4, input_size: int = 224):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        print("Folder %s created!" % data_dir)
    else:
        print("Folder %s already exists" % data_dir)
    
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(input_size + 32),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_set  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, test_loader

@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="eval"):
    model.eval()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    for images, targets in tqdm(loader, desc=f"Evaluating-{desc}", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets).item()
        top1, top5 = accuracy(outputs, targets)
        loss_meter.update(loss, images.size(0))
        top1_meter.update(top1, images.size(0))
        top5_meter.update(top5, images.size(0))

    print(f"[{desc}] Loss: {loss_meter.avg:.4f} | Top1: {top1_meter.avg:.2f}% | Top5: {top5_meter.avg:.2f}%")
    return loss_meter.avg, top1_meter.avg, top5_meter.avg



def vitis_ai_ptq(model: nn.Module,
                 calib_loader: DataLoader,
                 test_loader: DataLoader,
                 example_inputs: torch.Tensor,
                 device: torch.device,
                 export_dir: str = "quantize_result",
                 deploy: bool = False,
                 calib_batches: int = 200):
    assert torch_quantizer is not None, "pytorch_nndct (Vitis-AI) is not installed. Use the Vitis-AI PyTorch Docker."

    if not os.path.exists(export_dir):
        os.mkdir(export_dir)
        print("Folder %s created!" % export_dir)
    else:
        print("Folder %s already exists" % export_dir)

    model.to(device).eval()
    example_inputs = example_inputs.to(device)

    # 1) Calibration
    print("[VAI PTQ] Calibration …")
    quantizer = torch_quantizer("calib", model, (example_inputs,), output_dir=export_dir)
    quant_model = quantizer.quant_model
    seen = 0
    with torch.no_grad():
        for images, _ in tqdm(calib_loader, desc="Calibration", leave=False):
            images = images.to(device)
            _ = quant_model(images)
            seen += 1
            if seen >= calib_batches:  # limit for speed
                break
    # export calibration results
    quantizer.export_quant_config()

    # 2) Evaluate INT8 accuracy (simulation)
    print("[VAI PTQ] Evaluating quantized model (simulated int8) …")
    criterion = nn.CrossEntropyLoss().to(device)
    q_loss, q_top1, q_top5 = evaluate(quant_model, test_loader, criterion, device, desc="quant-sim")

    # 3) Export xmodel (test/deploy stage). Batch must be 1.
    print("[VAI PTQ] Exporting xmodel …")
    quantizer = torch_quantizer("test", model, (example_inputs,), output_dir=export_dir)
    test_qmodel = quantizer.quant_model
    
    with torch.no_grad():
        _ = test_qmodel(example_inputs)   # <-- one forward, batch=1
    
    quantizer.export_xmodel(output_dir=export_dir, deploy_check=deploy)
    print(f"[VAI PTQ] xmodel exported to: {export_dir}")

    return {"quant_top1": q_top1, "quant_top5": q_top5}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNet18 pruning→quantization on CIFAR-10")
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--input-size', type=int, default=224)

    parser.add_argument('--export-dir', type=str, default='./quantize_result')
    parser.add_argument('--deploy', action='store_true', help='enable deploy_check during export_xmodel')
    parser.add_argument('--calib-batches', type=int, default=100)

    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = cifar10_loaders(args.data_dir, args.batch_size, args.workers, args.input_size)

    criterion = nn.CrossEntropyLoss().to(device)

    model = torch.load('./best_pruned.pth',map_location="cpu")
    
    example_inputs = torch.randn(1, 3, args.input_size, args.input_size)

    stats = vitis_ai_ptq(model, train_loader, test_loader, example_inputs, device,
                             export_dir=args.export_dir, deploy=args.deploy, calib_batches=args.calib_batches)
    print(f"Quantized (sim) Top1={stats['quant_top1']:.2f} | Top5={stats['quant_top5']:.2f}")

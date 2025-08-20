import argparse,time,os
import torch,torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp
from pathlib import Path
import sys
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from model import Conv,ResBlock,build_backbone,FPN,YOLOv3,YoloHead,DetectLayer
from dataloader import Dataset,BasicTransform

@torch.no_grad()
def benchmark_fps(model, input_size=608, runs=50, warmup=10, device="cpu"):
    model.eval().to(device)
    x = torch.randn(1, 3, input_size, input_size, device=device)
    # warmup
    for _ in range(warmup):
        _ = model(x)
    torch.cuda.synchronize() if device.startswith("cuda") else None
    t0 = time.time()
    for _ in range(runs):
        _ = model(x)
    torch.cuda.synchronize() if device.startswith("cuda") else None
    dt = time.time() - t0
    fps = runs / dt
    return fps


def build_model(input_size, num_classes, anchors, model_type, ckpt_path):
    model = YOLOv3(input_size=input_size,
                   num_classes=num_classes,
                   anchors=anchors,
                   model_type=model_type,
                   pretrained=False)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model_state"] if "model_state" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=True)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    return model

import torch.nn as nn

def collect_ignored_convs(model, keep_stem=False, keep_stage_entry=False):
    ignored = set()
    for name, m in model.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue


        if name.endswith(".detect"):
            ignored.add(m)
            continue

        if ".res_block" in name and ".conv2.conv.0" in name:
            ignored.add(m)
            continue

        if keep_stem and name == "backbone.conv1.conv.0":
            ignored.add(m)
            continue

        if keep_stage_entry and (name.endswith(".res_block1.conv.conv.0")
                                 or name.endswith(".res_block2.conv.conv.0")
                                 or name.endswith(".res_block3.conv.conv.0")
                                 or name.endswith(".res_block4.conv.conv.0")
                                 or name.endswith(".res_block5.conv.conv.0")):
            ignored.add(m)
            continue

    return list(ignored)


def channel_prune(model, input_size=416, target_sparsity=0.5, stepwise=True):
    model.eval()
    example_inputs = torch.randn(1, 3, input_size, input_size)

    importance = tp.importance.BNScaleImportance()  
    ignored_layers = collect_ignored_convs(model)
    print(len(ignored_layers))
    
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=importance,
        iterative_steps=10 if stepwise else 1,
        pruning_ratio=target_sparsity,
        global_pruning=True,
        ignored_layers=ignored_layers,
    )


    print(f"[prune] target global channel sparsity = {target_sparsity}")
    pruner.step()
    return model

# -----------------------------
# BatchNorm re-calibration pass
# -----------------------------
@torch.no_grad()
def bn_recalibration(model, data_loader, iters=200):
    """
    Run a few hundred forward passes (no grad) in train mode to
    re-estimate BN running stats after structural changes.
    """
    model.train()
    i = 0
    for images, _ in data_loader:
        _ = model(images)
        i += 1
        if i >= iters:
            break
    model.eval()


def export_all(model, input_size, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model, os.path.join(out_dir, "yolov3_pruned_full.pt"))  # easiest
    model.eval()
    ex = torch.randn(1, 3, input_size, input_size)
    # TorchScript
    ts = torch.jit.trace(model, ex)
    ts.save(os.path.join(out_dir, "yolov3_pruned_script.pt"))
    # ONNX
    torch.onnx.export(model, ex, os.path.join(out_dir, "yolov3_pruned.onnx"),
                      opset_version=12, do_constant_folding=True,
                      input_names=["images"], output_names=["preds"])
    


def count_params(m):
    return sum(p.numel() for p in m.parameters())

def fps_report(fps_before: float, fps_after: float):
    delta = fps_after - fps_before
    speedup = fps_after / fps_before
    pct_fps = (speedup - 1.0) * 100.0
    lat_before = 1000.0 / fps_before
    lat_after  = 1000.0 / fps_after
    pct_lat = (1.0 - lat_after / lat_before) * 100.0
    print(f"FPS: {fps_before:.2f} -> {fps_after:.2f}  |  +{delta:.2f} FPS ({pct_fps:.2f}%), speedup {speedup:.3f}x")
    print(f"Latency: {lat_before:.2f} ms -> {lat_after:.2f} ms  |  {-pct_lat:.2f}% change" if pct_lat < 0
          else f"Latency reduced by {pct_lat:.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="path to trained yolov3 .pt")
    ap.add_argument("--input_size", type=int, default=416)
    ap.add_argument("--num_classes", type=int, default=20)
    ap.add_argument("--model_type", type=str, default="base")  # matches your naming
    ap.add_argument("--sparsity", type=float, default=0.4)
    ap.add_argument("--out", type=str, default="pruned_export")
    args = ap.parse_args()

    # Example COCO/VOC anchors parser (expect 9 numbers*2)
    anchors = [[0.248, 0.7237237 ],
        [0.36144578, 0.53      ],
        [0.42,       0.9306667 ],
        [0.456,      0.6858006 ],
        [0.488,      0.8168168 ],
        [0.6636637,  0.274     ],
        [0.806,      0.648     ],
        [0.8605263,  0.8736842 ],
        [0.944,      0.5733333 ]]

    model = build_model(args.input_size, args.num_classes, anchors, args.model_type, args.ckpt)

    if hasattr(model, "set_grid_xy"):
        model.set_grid_xy(args.input_size)
    FPS_before = benchmark_fps(model, args.input_size, runs=30)
    print(f"[pre] FPS (CPU, {args.input_size}) = {FPS_before:.2f}")
    def count_params(m): return sum(p.numel() for p in m.parameters())
    pre_params = count_params(model)
    pre_ch = model.head.detect_s.conv.conv[0].out_channels
    print("Params BEFORE:", pre_params, " | pre-detect3x3 out_ch BEFORE:", pre_ch)
    example_inputs = torch.randn(1, 3, args.input_size, args.input_size)

    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"MACs Before Pruning: {base_macs / 1e9:2f}G,{base_nparams/1e6:.2f}M")

    for p in model.parameters():
        p.requires_grad_(True)

    with torch.enable_grad():               
        pruned = channel_prune(
            model,                          
            input_size=args.input_size,
            target_sparsity=args.sparsity, 
            stepwise=True
        )

    post_params = count_params(model)
    post_ch = model.head.detect_s.conv.conv[0].out_channels
    print("Params AFTER :", post_params,  " | pre-detect3x3 out_ch AFTER :", post_ch)
    FPS_After = benchmark_fps(model, args.input_size, runs=30)

    print(f"[post] FPS (CPU, {args.input_size}) = {FPS_After:.2f}")
    print(f"Params after:  {post_params/1e6:.2f}M, reduction {(1 - post_params/pre_params)*100:.1f}%")
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"MACs Before Pruning: {base_macs / 1e9:2f}G,{base_nparams/1e6:.2f}M")
    fps_report(FPS_before,FPS_After)

    val_dataset = Dataset('/home/saurav/Desktop/Internship/ML-Internship-Saurav-Paudel/Paper_Implementation/ObjectDetection/UniYOLO/data/voc.yaml',phase = 'val')

    transform = BasicTransform(input_size = 416)
    val_dataset.load_transformer(transformer = transform)
    val_loader = DataLoader(val_dataset,batch_size = 1,shuffle = True,num_workers = 1,collate_fn = Dataset.collate_fn,pin_memory = True,)


    bn_recalibration(pruned, val_loader, iters=200)

    export_all(pruned, args.input_size, args.out)
    print(f"[done] exports saved to {args.out}")

    # model.eval()
    # ex = torch.randn(1,3,args.input_size,args.input_size)
    # DG = tp.DependencyGraph().build_dependency(model, example_inputs=ex)

    # conv = model.head.detect_s.conv.conv[0]      # pre-detect 3x3
    # keep_last = list(range(conv.out_channels-8, conv.out_channels))
    # plan = DG.get_pruning_group(conv, tp.prune_conv_out_channels, idxs=keep_last)
    # print("plan ops:", len(plan))
    # plan.exec()                                  # APPLY
    # print("pre-detect3x3 out_ch NOW:", conv.out_channels)



if __name__ == "__main__":
    main()



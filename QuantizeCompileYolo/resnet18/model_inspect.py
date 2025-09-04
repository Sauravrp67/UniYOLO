import torch
from pytorch_nndct.apis import Inspector
import torchvision.models as models 
import os
import torch_pruning as tp
import torch.nn as nn
# model = torch.load('./best_pruned.pth',map_location='cpu')

def build_resnet18(num_classes=10, pretrained=True):
    """Torchvision resnet18 adapted for CIFAR-10."""
    try:
        # torchvision>=0.13 style
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=pretrained)

    # Replace final FC for 10 classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_resnet18()
dummy_input = torch.randn(1,3,224,224)

base_macs, base_params = tp.utils.count_ops_and_params(model, dummy_input.to(device))
print(f"[Pruning] Baseline: MACs={base_macs/1e6:.2f}M | Params={base_params/1e6:.2f}M")

state_dict = torch.load('/workspace/QuantizeCompileYolo/resnet18/best_float.pth',map_location="cpu")

model.load_state_dict(state_dict=state_dict['state_dict'],strict = True)

target = "0x101000056010407"

inspector = Inspector(target)


path = "./inspect_float"

# check whether directory already exists
if not os.path.exists(path):
  os.mkdir(path)
  print("Folder %s created!" % path)
else:
  print("Folder %s already exists" % path)

inspector.inspect(model,(dummy_input,),device = device,output_dir = path,verbose_level = 2,image_format = "png")
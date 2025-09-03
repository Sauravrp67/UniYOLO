import torch
from pytorch_nndct.apis import Inspector
from torchvision.models.resnet import resnet18
import os
import torch_pruning as tp

model = torch.load('./best_pruned.pth',map_location='cpu')
dummy_input = torch.randn(1,3,224,224)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_macs, base_params = tp.utils.count_ops_and_params(model, dummy_input.to(device))
print(f"[Pruning] Baseline: MACs={base_macs/1e6:.2f}M | Params={base_params/1e6:.2f}M")


target = "0x101000056010407"

inspector = Inspector(target)


path = "./inspect"

# check whether directory already exists
if not os.path.exists(path):
  os.mkdir(path)
  print("Folder %s created!" % path)
else:
  print("Folder %s already exists" % path)

inspector.inspect(model,(dummy_input,),device = device,output_dir = path,verbose_level = 2,image_format = "png")
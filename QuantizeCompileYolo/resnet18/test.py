import torch
import torch_pruning as tp

model = torch.load('./best_pruned.pth',map_location="cpu",weights_only=False)

example_input = torch.randn(1,3,224,224)

macs, params = tp.utils.count_ops_and_params(model, example_input.to('cpu'))
print(f"MACs={macs/1e6:.2f}M | Params={params/1e6:.2f}M")
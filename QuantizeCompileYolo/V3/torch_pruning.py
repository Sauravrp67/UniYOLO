# prune_tiny_sequential.py
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_nndct import get_pruning_runner

# ----------------------------
# 1) Tiny Sequential Conv Net
# ----------------------------
class TinyConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),   # -> [B,32,1,1]
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

# -----------------------------------------
# 2) Tiny random dataset for search/calib
# -----------------------------------------
class RandomDataset(Dataset):
    def __init__(self, n=256, num_classes=10, img_size=64):
        self.n = n
        self.num_classes = num_classes
        self.c, self.h, self.w = 3, img_size, img_size
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        x = torch.randn(self.c, self.h, self.w)
        y = torch.randint(0, self.num_classes, (1,)).item()
        return x, y

def tiny_calib_fn(model, loader, number_forward=64):
    """Run a few forward passes (no grad) to let BN adapt; used during search."""
    model.train()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            x = x.to(next(model.parameters()).device)
            _ = model(x)
            if i >= number_forward:
                break

def tiny_eval_fn(model, loader):
    """Return a scalar score; search just needs any monotonic metric."""
    model.eval()
    score = 0.0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(next(model.parameters()).device)
            _ = model(x)
            score += 1.0
            if i >= 10:
                break
    # Return tensor/float; pruning runner accepts both
    return torch.tensor(score, device=next(model.parameters()).device)

# ----------------------------
# 3) Wire everything together
# ----------------------------
if __name__ == "__main__":
    # Make sure working dir exists for .vai artifacts
    os.makedirs(".vai", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyConvNet(num_classes=10).to(device).train()

    # Dummy input signature for graph tracing
    input_signature = torch.randn(1, 3, 64, 64, device=device)

    # Build pruning runner (one_step so we can search then slim-prune)
    pruning_runner = get_pruning_runner(model, input_signature, method="one_step")

    # Small random loaders for search/calibration
    train_loader = DataLoader(RandomDataset(n=256), batch_size=32, shuffle=True)
    eval_loader  = DataLoader(RandomDataset(n=64),  batch_size=32, shuffle=False)

    # -------------------------
    # 3a) SEARCH (required for slim)
    # -------------------------
    removal_ratio = 0.5       # prune ~50% channels
    channel_divisible = 2     # keep channels divisible by 2
    num_subnet = 8            # quick search

    pruning_runner.search(
        gpus=[0] if torch.cuda.is_available() else [],
        calibration_fn=tiny_calib_fn,
        eval_fn=tiny_eval_fn,
        num_subnet=num_subnet,
        removal_ratio=removal_ratio,
        calib_args=(train_loader,),
        eval_args=(eval_loader,)
    )


    slim_model = pruning_runner.prune(
        removal_ratio=removal_ratio,
        mode="slim",
        index=None,                   # let runner pick best subnet
        channel_divisible=channel_divisible
    ).to(device)

    # Sanity forward
    slim_model.train()
    x = torch.randn(4, 3, 64, 64, device=device)
    out = slim_model(x)
    print("Sanity output shape:", tuple(out.shape))  # expect [4, 10]

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(slim_model.parameters(), lr=1e-3, momentum=0.9)

    for step in range(5):
        xb = torch.randn(32, 3, 64, 64, device=device)
        yb = torch.randint(0, 10, (32,), device=device)
        optimizer.zero_grad()
        loss = criterion(slim_model(xb), yb)
        loss.backward()
        optimizer.step()
        print(f"[Fine-tune] step {step}: loss={loss.item():.4f}")

    torch.save(slim_model.state_dict(), "tinyconv_pruned.pth")
    if hasattr(slim_model, "slim_state_dict"):
        torch.save(slim_model.slim_st)

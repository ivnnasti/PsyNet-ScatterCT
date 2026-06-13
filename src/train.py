import os
import sys
import json
import time
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

torch.set_num_threads(min(4, os.cpu_count() or 4))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import get_datasets
from model import ResUNet
from loss import total_loss
from runtime_config import get_config

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = get_config(_root)
DATA_DIR = cfg["data_dir"]
EPOCHS = cfg["epochs"]
BS = cfg["bs"]
LR = cfg["lr"]
CH = cfg["ch"]
STEP = cfg["step"]
SIZE = cfg["size"]
MAX_VOLS = cfg["max_vols"]
VAL_BS = cfg["val_bs"]
_out = cfg["out"]
os.makedirs(_out, exist_ok=True)

print(f"config: profile={cfg['profile']} epochs={EPOCHS} bs={BS} lr={LR} ch={CH} step={STEP} size={SIZE} max_vols={MAX_VOLS or 'all'}")
print("loading data...")
train_ds, val_ds = get_datasets(DATA_DIR, cbct_res=cfg["cbct_res"], step=STEP, size=SIZE, max_vols=MAX_VOLS)
train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True,  num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=VAL_BS, shuffle=False, num_workers=0)
print(f"train batches: {len(train_dl)}  val batches: {len(val_dl)}")

device = torch.device("cpu")
model = ResUNet(ch=CH).to(device)
params = sum(p.numel() for p in model.parameters())
print(f"model params: {params:,}")

print("warmup...")
_dummy = torch.zeros(1, 1, SIZE, SIZE)
with torch.no_grad():
    _ = model(_dummy)
print("warmup done")

opt   = Adam(model.parameters(), lr=LR)
sched = CosineAnnealingLR(opt, T_max=EPOCHS)

best_val = float("inf")
log = {"train": [], "val": []}

for ep in range(1, EPOCHS + 1):
    model.train()
    tr = 0.0
    t0 = time.time()
    n_train = len(train_dl)
    step_log = max(1, min(50, n_train))
    print(f"ep {ep}/{EPOCHS} start")
    for i, (x, y) in enumerate(train_dl, 1):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = total_loss(model(x), y)
        loss.backward()
        opt.step()
        tr += loss.item()
        if i % step_log == 0 or i == n_train:
            elapsed = time.time() - t0
            eta = elapsed / i * (n_train - i)
            print(f"  ep {ep}/{EPOCHS}  batch {i}/{n_train}  loss {loss.item():.4f}  eta {eta:.0f}s", flush=True)
    tr /= n_train

    model.eval()
    vl = 0.0
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            vl += total_loss(model(x), y).item()
    vl /= len(val_dl)
    sched.step()

    ep_time = time.time() - t0
    log["train"].append(tr)
    log["val"].append(vl)
    print(f"ep {ep}/{EPOCHS} | train {tr:.4f} | val {vl:.4f} | {ep_time:.0f}s")

    if vl < best_val:
        best_val = vl
        torch.save(model.state_dict(), os.path.join(_out, "best_model.pth"))
        print(f"  saved best_model.pth (val {vl:.4f})")

with open(os.path.join(_out, "train_log.json"), "w") as f:
    json.dump(log, f)

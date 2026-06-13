import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as sk_ssim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import get_datasets
from model import ResUNet
from runtime_config import get_config

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = get_config(_root)
DATA_DIR = cfg["data_dir"]
CH = cfg["ch"]
STEP = cfg["step"]
SIZE = cfg["size"]
MAX_VOLS = cfg["max_vols"]
_out = cfg["out"]

_, val_ds = get_datasets(DATA_DIR, cbct_res=cfg["cbct_res"], step=STEP, size=SIZE, max_vols=MAX_VOLS)
val_dl = DataLoader(val_ds, batch_size=cfg["val_bs"], shuffle=False, num_workers=0)

device = torch.device("cpu")
model = ResUNet(ch=CH).to(device)
model.load_state_dict(torch.load(os.path.join(_out, "best_model.pth"), map_location=device))
model.eval()

rmse_list, ssim_list = [], []
with torch.no_grad():
    for x, y in val_dl:
        pred = model(x.to(device)).cpu().numpy()[0, 0]
        gt = y.numpy()[0, 0]
        rmse_list.append(float(np.sqrt(np.mean((pred - gt) ** 2))))
        ssim_list.append(float(sk_ssim(pred, gt, data_range=2.0)))

rmse = float(np.mean(rmse_list))
ssim = float(np.mean(ssim_list))

os.makedirs(_out, exist_ok=True)
with open(os.path.join(_out, "metrics.json"), "w") as f:
    json.dump({"rmse": rmse, "ssim": ssim}, f)

print(f"rmse: {rmse:.4f} | ssim: {ssim:.4f}")

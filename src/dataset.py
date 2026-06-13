import os
import re
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def load_nii(path):
    return nib.load(path).get_fdata(dtype=np.float32)


def norm_cbct(x):
    return (x.astype(np.float32) / 32767.5) - 1.0


def norm_ct(x):
    x = np.clip(x, -1024.0, 1500.0)
    return (x - 238.0) / 1262.0


def norm(x):
    return norm_cbct(x)


def _idx(path):
    m = re.search(r'(\d+)\.nii', os.path.basename(path))
    return int(m.group(1)) if m else -1


def _resize(arr, size):
    if size is None:
        return arr
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode='bilinear', align_corners=False)
    return t[0, 0].numpy()


class SliceDataset(Dataset):
    def __init__(self, cbct_paths, ct_paths, step=1, size=None, tag=""):
        cbct_list, ct_list = [], []
        n_vols = len(cbct_paths)
        for i, (cp, tp) in enumerate(zip(cbct_paths, ct_paths)):
            print(f"\r  {tag} loading volume {i+1}/{n_vols} ...", end="", flush=True)
            cbct = norm_cbct(load_nii(cp))
            ct   = norm_ct(load_nii(tp))
            n = min(cbct.shape[2], ct.shape[2])
            for j in range(0, n, step):
                cbct_list.append(_resize(cbct[:, :, j].copy(), size))
                ct_list.append(_resize(ct[:, :, j].copy(), size))
        self.cbct = np.stack(cbct_list)
        self.ct   = np.stack(ct_list)
        print(f"\r  {tag} {len(self.cbct)} slices from {n_vols} volumes{' ' * 20}")

    def __len__(self):
        return len(self.cbct)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.cbct[idx]).unsqueeze(0)
        y = torch.from_numpy(self.ct[idx]).unsqueeze(0)
        return x, y


def get_datasets(data_dir, cbct_res=None, step=1, size=None, max_vols=0):
    if cbct_res is None:
        cbct_res = os.environ.get("CBCT_RES", "128")
    cbct_dir = os.path.join(data_dir, "TRAINCBCTSimulated", cbct_res)
    ct_dir = os.path.join(data_dir, "TRAINCTAlignedToCBCT")
    cbct_all = sorted(glob.glob(os.path.join(cbct_dir, "*.nii*")), key=_idx)
    ct_all = sorted(glob.glob(os.path.join(ct_dir, "*.nii*")), key=_idx)
    cbct_map = {_idx(p): p for p in cbct_all}
    ct_map = {_idx(p): p for p in ct_all}
    common = sorted(set(cbct_map) & set(ct_map))
    assert len(common) > 0, f"no matched pairs in {cbct_dir} / {ct_dir}"
    if max_vols > 0:
        common = common[:max_vols]
    cbct_files = [cbct_map[i] for i in common]
    ct_files = [ct_map[i] for i in common]
    split = max(1, int(len(common) * 0.8))
    train_ds = SliceDataset(cbct_files[:split], ct_files[:split], step=step, size=size, tag="train")
    val_ds = SliceDataset(cbct_files[split:], ct_files[split:], step=step, size=size, tag="val")
    return train_ds, val_ds

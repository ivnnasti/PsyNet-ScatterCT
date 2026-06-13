import os


PROFILES = {
    "fast": {
        "epochs": 5,
        "bs": 8,
        "lr": 1e-3,
        "ch": 4,
        "step": 16,
        "size": 96,
        "val_bs": 16,
    },
    "balanced": {
        "epochs": 8,
        "bs": 8,
        "lr": 1e-3,
        "ch": 4,
        "step": 8,
        "size": 96,
        "val_bs": 16,
    },
    "full": {
        "epochs": 20,
        "bs": 4,
        "lr": 1e-3,
        "ch": 8,
        "step": 4,
        "size": 128,
        "val_bs": 8,
    },
}


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_float(name, default):
    return float(os.environ.get(name, default))


def get_profile():
    profile = os.environ.get("PROFILE", "balanced").strip().lower()
    return profile if profile in PROFILES else "balanced"


def get_config(root):
    profile = get_profile()
    base = PROFILES[profile]
    return {
        "profile": profile,
        "data_dir": os.environ.get("DATA_DIR", os.path.join(root, "data")),
        "epochs": _env_int("EPOCHS", base["epochs"]),
        "bs": _env_int("BS", base["bs"]),
        "lr": _env_float("LR", base["lr"]),
        "ch": _env_int("CH", base["ch"]),
        "step": _env_int("STEP", base["step"]),
        "size": _env_int("SIZE", base["size"]),
        "val_bs": _env_int("VAL_BS", base["val_bs"]),
        "max_vols": _env_int("MAX_VOLS", 0),
        "cbct_res": os.environ.get("CBCT_RES", "128"),
        "out": os.path.join(root, "output"),
    }
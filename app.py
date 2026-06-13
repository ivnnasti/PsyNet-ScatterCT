import os
import sys
import json
import base64
import io
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import torch
from flask import Flask, request, jsonify, send_file, render_template

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from model import ResUNet
from dataset import norm_cbct
from runtime_config import get_config

app = Flask(__name__)

_cfg = get_config(os.path.dirname(os.path.abspath(__file__)))
CH = _cfg["ch"]
MODEL_PATH = "output/best_model.pth"

_model = None


def get_model():
    global _model
    if _model is None and os.path.exists(MODEL_PATH):
        m = ResUNet(ch=CH)
        m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        m.eval()
        _model = m
    return _model


def slice_to_png(arr):
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(arr, cmap="gray", origin="lower")
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/metrics")
def metrics():
    path = "output/metrics.json"
    if not os.path.exists(path):
        return jsonify({"error": "не найдено"})
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/plot")
def plot():
    path = "output/training_curves.png"
    if not os.path.exists(path):
        return ("", 404)
    return send_file(path, mimetype="image/png")


@app.route("/predict", methods=["POST"])
def predict():
    model = get_model()
    if model is None:
        return jsonify({"error": "модель не обучена"})

    f = request.files.get("file")
    if f is None:
        return jsonify({"error": "файл не передан"})

    suffix = ".nii.gz" if f.filename.endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        data = nib.load(tmp_path).get_fdata(dtype=np.float32)
    finally:
        os.unlink(tmp_path)

    data = norm_cbct(data)
    mid = data.shape[2] // 2
    slc = data[:, :, mid]

    x = torch.from_numpy(slc).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        pred = model(x).numpy()[0, 0]

    return jsonify({
        "cbct": slice_to_png(slc),
        "pred": slice_to_png(pred),
    })


if __name__ == "__main__":
    app.run(debug=False)

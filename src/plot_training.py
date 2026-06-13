import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
with open(os.path.join(_root, "output", "train_log.json")) as f:
    log = json.load(f)

epochs = range(1, len(log["train"]) + 1)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, log["train"], label="обучение")
ax.plot(epochs, log["val"], label="валидация")
ax.set_xlabel("эпоха")
ax.set_ylabel("потери")
ax.legend()
fig.tight_layout()
out_path = os.path.join(_root, "output", "training_curves.png")
fig.savefig(out_path, dpi=150)
plt.close(fig)
print("saved output/training_curves.png")

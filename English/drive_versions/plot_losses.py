import re, sys, os, yaml
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

cfg = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)
out_dir = sys.argv[2]
log_dir = cfg["train"]["path"]["log_path"]
os.makedirs(out_dir, exist_ok=True)

def parse(path):
    rows = {}
    for line in open(path):
        m = re.search(r"Step (\d+)", line)
        if m:
            rows[int(m.group(1))] = {k.strip(): float(v) for k, v in
                                     re.findall(r"([A-Za-z ]+?) Loss: ([0-9.eE+-]+)", line)}
    return rows

def plot(rows, title, fname, smooth=1):
    steps = sorted(rows)
    plt.figure(figsize=(9, 5))
    for k in ["Total", "Mel", "Mel PostNet", "Pitch", "Energy", "Duration"]:
        y = np.array([rows[s].get(k, np.nan) for s in steps])
        if smooth > 1:
            y = np.convolve(y, np.ones(smooth) / smooth, mode="valid"); x = steps[smooth - 1:]
        else:
            x = steps
        plt.plot(x, y, label=k, lw=2 if k == "Total" else 1)
    plt.xlabel("step"); plt.ylabel("loss"); plt.title(title); plt.legend(); plt.grid(alpha=.3)
    plt.savefig(os.path.join(out_dir, fname), dpi=200, bbox_inches="tight"); plt.close()

tr = parse(os.path.join(log_dir, "train", "log.txt"))
va = parse(os.path.join(log_dir, "val", "log.txt"))
plot(tr, "Training loss", "training_loss.png", smooth=5)
plot(va, "Validation loss", "validation_loss.png")
plt.figure(figsize=(8, 4.5))
plt.plot(sorted(va), [va[s]["Total"] for s in sorted(va)], label="validation total")
ts = sorted(tr); plt.plot(ts[4:], np.convolve([tr[s]["Total"] for s in ts], np.ones(5) / 5, "valid"), label="training total (smoothed)")
plt.xlabel("step"); plt.ylabel("total loss"); plt.legend(); plt.grid(alpha=.3)
plt.savefig(os.path.join(out_dir, "train_vs_val_total.png"), dpi=200, bbox_inches="tight")
print("train points:", len(tr), "val points:", len(va))
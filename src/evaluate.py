from __future__ import annotations
import os, json, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from .config import COMMANDS
from .data import SubsetSC, filter_commands
from .features import waveform_to_logmelspec
from .model import SmallKWS, model_size_bytes


def collate_fn(batch, label2idx):
    xs, ys = [], []
    for waveform, _, label, _, _ in batch:
        x = waveform_to_logmelspec(waveform)
        xs.append(x)
        ys.append(label2idx[label])
    return torch.stack(xs, 0), torch.tensor(ys, dtype=torch.long)


def human_size(num_bytes: int) -> str:
    kb = num_bytes / 1024
    mb = kb / 1024
    if mb >= 1:
        return f"{mb:.2f} MB"
    return f"{kb:.2f} KB"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--model_path", default="models/model.pt")
    ap.add_argument("--out_dir", default="models")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--latency_samples", type=int, default=200)
    args = ap.parse_args()

    ckpt = torch.load(args.model_path, map_location="cpu")
    label2idx = ckpt["label2idx"]

    model = SmallKWS(num_classes=len(COMMANDS))
    model.load_state_dict(ckpt["state_dict"])
    model.to(args.device)
    model.eval()

    test_ds = SubsetSC(args.data_root, "testing", download=True)
    test_ds = filter_commands(test_ds, COMMANDS, max_per_class=None)

    loader = DataLoader(
        test_ds, batch_size=args.batch, shuffle=False,
        collate_fn=lambda b: collate_fn(b, label2idx), num_workers=0
    )

    # ===== Accuracy =====
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(args.device)
            logits = model(X)
            pred = logits.argmax(dim=1).cpu().numpy()
            y_true.extend(y.numpy().tolist())
            y_pred.extend(pred.tolist())

    acc = accuracy_score(y_true, y_pred)

    # ===== Latency (ms) =====
    lat_list = []
    with torch.no_grad():
        for i, (X, _) in enumerate(loader):
            if i * X.size(0) >= args.latency_samples:
                break
            for j in range(X.size(0)):
                x1 = X[j:j+1].to(args.device)
                t0 = time.perf_counter()
                _ = model(x1)
                t1 = time.perf_counter()
                lat_list.append((t1 - t0) * 1000.0)

    latency_ms = float(np.mean(lat_list)) if lat_list else 0.0

    # ===== Model size =====
    file_size = os.path.getsize(args.model_path)       # на диску
    param_bytes = model_size_bytes(model)              # параметри в памʼяті

    # ===== Console output (те, що тобі треба) =====
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Latency:  {latency_ms:.2f} ms")
    print(f"Model size (file):   {human_size(file_size)}")
    print(f"Model size (params): {human_size(param_bytes)}")

    # ===== Save metrics.json =====
    metrics = {
        "accuracy": float(acc),
        "latency_ms_avg": latency_ms,
        "model_file_bytes": int(file_size),
        "model_param_bytes": int(param_bytes),
        "commands": COMMANDS,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Saved metrics.json")


if __name__ == "__main__":
    main()

from __future__ import annotations
import os, json, random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from .config import COMMANDS, BATCH_SIZE_DEFAULT, EPOCHS_DEFAULT, LR_DEFAULT, SEED
from .data import SubsetSC, filter_commands
from .features import waveform_to_logmelspec
from .model import SmallKWS

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def collate_fn(batch, label2idx):
    xs, ys = [], []
    for waveform, _, label, _, _ in batch:
        x = waveform_to_logmelspec(waveform)   # (1, n_mels, time)
        xs.append(x)
        ys.append(label2idx[label])
    X = torch.stack(xs, dim=0)                 # (B, 1, n_mels, time)
    y = torch.tensor(ys, dtype=torch.long)
    return X, y

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE_DEFAULT)
    ap.add_argument("--lr", type=float, default=LR_DEFAULT)
    ap.add_argument("--max_per_class", type=int, default=5000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_dir", default="models")
    args = ap.parse_args()

    seed_all(SEED)
    os.makedirs(args.out_dir, exist_ok=True)

    label2idx = {c: i for i, c in enumerate(COMMANDS)}

    train_ds = SubsetSC(args.data_root, "training", download=True)
    test_ds  = SubsetSC(args.data_root, "testing",  download=True)

    train_ds = filter_commands(train_ds, COMMANDS, max_per_class=args.max_per_class)
    test_ds  = filter_commands(test_ds,  COMMANDS, max_per_class=None)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        collate_fn=lambda b: collate_fn(b, label2idx), num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch, shuffle=False,
        collate_fn=lambda b: collate_fn(b, label2idx), num_workers=0
    )

    model = SmallKWS(num_classes=len(COMMANDS)).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(args.device), y.to(args.device)
            opt.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * X.size(0)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(args.device), y.to(args.device)
                pred = model(X).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()

        print(f"Epoch {epoch}/{args.epochs} | loss={total_loss/len(train_ds):.4f} | test_acc={correct/total:.4f}")

    model_path = os.path.join(args.out_dir, "model.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "commands": COMMANDS,
        "label2idx": label2idx,
    }, model_path)

    with open(os.path.join(args.out_dir, "label2idx.json"), "w", encoding="utf-8") as f:
        json.dump(label2idx, f, ensure_ascii=False, indent=2)

    print(f"Saved: {model_path}")

if __name__ == "__main__":
    main()

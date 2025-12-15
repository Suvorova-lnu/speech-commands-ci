from __future__ import annotations

import argparse
import sys
import torch

from src.model import SmallKWS
from src.config import COMMANDS
from src.features import load_audio, waveform_to_logmelspec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/model.pt")
    ap.add_argument("--wav", default="assets/stable_samples/go.wav")
    ap.add_argument("--expected", default="go")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min_prob", type=float, default=0.30)
    args = ap.parse_args()

    ckpt = torch.load(args.model_path, map_location="cpu")

    model = SmallKWS(num_classes=len(COMMANDS))
    model.load_state_dict(ckpt["state_dict"])
    model.to(args.device)
    model.eval()

    wav = load_audio(args.wav)
    x = waveform_to_logmelspec(wav).unsqueeze(0).to(args.device)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu()[0]

    pred_idx = int(torch.argmax(probs).item())
    pred_label = COMMANDS[pred_idx]
    pred_prob = float(probs[pred_idx].item())

    print(f"Stable sample: {args.wav}")
    print(f"Expected: {args.expected}")
    print(f"Pred: {pred_label}")
    print(f"Prob: {pred_prob:.4f}")

    if pred_label != args.expected or pred_prob < args.min_prob:
        print("VERIFY FAILED")
        return 1

    print("VERIFY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

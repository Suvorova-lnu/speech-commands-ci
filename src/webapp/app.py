from __future__ import annotations
import os, json, time, random, tempfile
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify

from ..config import COMMANDS
from ..model import SmallKWS
from ..features import waveform_to_logmelspec, load_audio
from ..data import SubsetSC, filter_commands, label_from_path

APP_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, "..", ".."))
MODELS_DIR = os.path.join(ROOT_DIR, "models")

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(MODELS_DIR, "model.pt"))
METRICS_PATH = os.environ.get("METRICS_PATH", os.path.join(MODELS_DIR, "metrics.json"))
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(ROOT_DIR, "data"))

app = Flask(__name__)

_ckpt = torch.load(MODEL_PATH, map_location="cpu")
model = SmallKWS(num_classes=len(COMMANDS))
model.load_state_dict(_ckpt["state_dict"])
model.eval()

metrics: dict = {}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

# ---------- Pretty formatting helpers ----------

def human_size(num_bytes: int) -> str:
    kb = num_bytes / 1024.0
    mb = kb / 1024.0
    return f"{mb:.2f} MB" if mb >= 1 else f"{kb:.2f} KB"

def make_global_metrics_pretty(m: dict) -> dict:
    out = {}
    if "accuracy" in m:
        out["accuracy_percent"] = f"{float(m['accuracy']) * 100:.2f}%"
    if "latency_ms_avg" in m:
        out["latency_ms_avg_str"] = f"{float(m['latency_ms_avg']):.2f} ms"
    if "model_file_bytes" in m:
        out["model_size_file_str"] = human_size(int(m["model_file_bytes"]))
    if "model_param_bytes" in m:
        out["model_size_params_str"] = human_size(int(m["model_param_bytes"]))
    return out

# ---------- Inference ----------

def predict_waveform(waveform: torch.Tensor) -> dict:
    x = waveform_to_logmelspec(waveform).unsqueeze(0)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(x)
    t1 = time.perf_counter()

    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    latency_ms = float((t1 - t0) * 1000.0)

    return {
        "pred": COMMANDS[pred_idx],
        "probs": {COMMANDS[i]: float(probs[i]) for i in range(len(COMMANDS))},
        "latency_ms": latency_ms,
        "latency_ms_str": f"{latency_ms:.2f} ms",
    }

@app.get("/")
def index():
    # На сторінці можна показувати як "raw", так і "pretty"
    return render_template(
        "index.html",
        commands=COMMANDS,
        metrics=metrics
    )

@app.post("/predict_text")
def predict_text():
    label = request.json.get("label")
    if label not in COMMANDS:
        return jsonify({"error": "label must be one of commands"}), 400

    ds = SubsetSC(DATA_ROOT, "testing", download=True)
    ds = filter_commands(ds, COMMANDS, max_per_class=None)

    candidates = [p for p in ds._walker if label_from_path(p) == label]
    if not candidates:
        return jsonify({"error": "no samples for label"}), 404

    path = random.choice(candidates)
    idx = ds._walker.index(path)
    waveform, _, _, _, _ = ds[idx]

    out = predict_waveform(waveform)
    out["expected"] = label

    # raw + pretty
    out["global_metrics"] = metrics
    out["global_metrics_pretty"] = make_global_metrics_pretty(metrics)
    return jsonify(out)

@app.post("/predict_audio")
def predict_audio():
    if "file" not in request.files:
        return jsonify({"error": "missing file"}), 400

    f = request.files["file"]

    # ✅ Windows-friendly temp file
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    try:
        f.save(path)
        wav = load_audio(path)
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

    out = predict_waveform(wav)

    # raw + pretty
    out["global_metrics"] = metrics
    out["global_metrics_pretty"] = make_global_metrics_pretty(metrics)
    return jsonify(out)

@app.get("/metrics")
def get_metrics():
    return jsonify({
        "global_metrics": metrics,
        "global_metrics_pretty": make_global_metrics_pretty(metrics)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

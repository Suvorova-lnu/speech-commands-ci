from __future__ import annotations
import os, json, time, random
import torch
import numpy as np

from .config import COMMANDS
from .features import load_audio, waveform_to_logmelspec
from .data import SubsetSC, filter_commands, label_from_path
from .model import SmallKWS, model_size_bytes


def human_size(num_bytes: int) -> str:
    kb = num_bytes / 1024
    mb = kb / 1024
    if mb >= 1:
        return f"{mb:.2f} MB"
    return f"{kb:.2f} KB"


def load_global_metrics(metrics_path: str) -> dict:
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def infer_one(model: torch.nn.Module, x: torch.Tensor, device: str):
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(x.to(device))
    t1 = time.perf_counter()

    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    pred_label = COMMANDS[pred_idx]
    latency_ms = float((t1 - t0) * 1000.0)
    return pred_label, probs, latency_ms


def pick_random_test_sample(data_root: str, expected_label: str):
    ds = SubsetSC(data_root, "testing", download=True)
    ds = filter_commands(ds, COMMANDS, max_per_class=None)

    candidates = [p for p in ds._walker if label_from_path(p) == expected_label]
    if not candidates:
        return None

    path = random.choice(candidates)
    wav = load_audio(path)
    x = waveform_to_logmelspec(wav).unsqueeze(0)  # (1, 1, n_mels, time)
    return path, x


def print_required_metrics(acc: float | None, latency_ms: float, model_path: str, model: torch.nn.Module):
    file_size = os.path.getsize(model_path)
    param_bytes = model_size_bytes(model)

    if acc is None:
        print("Accuracy: (нема) → запусти evaluate, щоб створився models/metrics.json")
    else:
        print(f"Accuracy: {acc * 100:.2f}%")

    print(f"Latency:  {latency_ms:.2f} ms")
    print(f"Model size (file):   {human_size(file_size)}")
    print(f"Model size (params): {human_size(param_bytes)}")


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    tail = " [Y/n]" if default else " [y/N]"
    s = input(prompt + tail + ": ").strip().lower()
    if not s:
        return default
    return s in ("y", "yes", "так", "t")


def menu_loop(model, acc, args):
    print("\n=== Console Inference Menu ===")
    print("1) Ввести команду текстом (go/up/right/stop) → випадковий тестовий файл")
    print("2) Інференс з WAV файлу (ввести шлях)")
    print("q) Вийти")
    print("==============================")

    while True:
        choice = input("\nОбери режим (1/2 або q): ").strip().lower()
        if choice in ("q", "quit", "exit"):
            print("Bye!")
            break

        show_probs = ask_yes_no("Показувати probabilities таблицю?", default=False)

        # 1) text command loop
        if choice == "1":
            print("\nВводь: right/stop/up/go. Вихід: q\n")
            while True:
                cmd = input("Enter command: ").strip().lower()
                if cmd in ("q", "quit", "exit"):
                    print("Вихід у меню.")
                    break
                if cmd not in COMMANDS:
                    print(f"❌ '{cmd}' не з команд. Дозволено: {', '.join(COMMANDS)}")
                    continue

                picked = pick_random_test_sample(args.data_root, cmd)
                if picked is None:
                    print("Не знайдено прикладів у тесті для цього класу.")
                    continue

                path, x = picked
                pred, probs, latency_ms = infer_one(model, x, args.device)

                print("\n--- RESULT ---")
                print_required_metrics(acc, latency_ms, args.model_path, model)
                print(f"Expected:  {cmd}")
                print(f"Predicted: {pred}")
                print(f"Sample:    {path}")
                if show_probs:
                    print("Probabilities:")
                    for i, c in enumerate(COMMANDS):
                        print(f"  {c:>5}: {float(probs[i]):.4f}")
                print("------------\n")
            continue

        # 2) wav file
        if choice == "2":
            wav_path = input('Введи шлях до WAV (наприклад C:\\Users\\...\\test.wav): ').strip().strip('"')
            if not wav_path:
                print("Шлях порожній.")
                continue
            if not os.path.exists(wav_path):
                print("❌ Файл не знайдено. Перевір шлях.")
                continue

            wav = load_audio(wav_path)
            x = waveform_to_logmelspec(wav).unsqueeze(0)
            pred, probs, latency_ms = infer_one(model, x, args.device)

            print("\n--- RESULT ---")
            print_required_metrics(acc, latency_ms, args.model_path, model)
            print(f"Predicted: {pred}")
            print(f"File:      {wav_path}")
            if show_probs:
                print("Probabilities:")
                for i, c in enumerate(COMMANDS):
                    print(f"  {c:>5}: {float(probs[i]):.4f}")
            print("------------\n")
            continue

        print("Невірний вибір. Введи 1, 2 або q.")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/model.pt")
    ap.add_argument("--metrics_path", default="models/metrics.json")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--device", default="cpu")

    # якщо запускати з аргументами теж можна (не обов’язково)
    ap.add_argument("--audio", default=None)
    ap.add_argument("--show_probs", action="store_true")
    args = ap.parse_args()

    ckpt = torch.load(args.model_path, map_location="cpu")
    model = SmallKWS(num_classes=len(COMMANDS))
    model.load_state_dict(ckpt["state_dict"])
    model.to(args.device).eval()

    global_metrics = load_global_metrics(args.metrics_path)
    acc = global_metrics.get("accuracy", None)

    # якщо передали --audio, зробимо разовий інференс з файлу
    if args.audio:
        wav = load_audio(args.audio)
        x = waveform_to_logmelspec(wav).unsqueeze(0)
        pred, probs, latency_ms = infer_one(model, x, args.device)

        print_required_metrics(acc, latency_ms, args.model_path, model)
        print(f"Predicted: {pred}")
        if args.show_probs:
            for i, c in enumerate(COMMANDS):
                print(f"  {c:>5}: {float(probs[i]):.4f}")
        return

    # інакше показуємо меню
    menu_loop(model, acc, args)


if __name__ == "__main__":
    main()

from __future__ import annotations
import os
from typing import List
import torchaudio

from .config import COMMANDS
from .features import load_audio

class SubsetSC(torchaudio.datasets.SPEECHCOMMANDS):
    """SpeechCommands with official splits (training/validation/testing),
    but WAV loading is done via SciPy to avoid TorchCodec requirement on Windows.
    """
    def __init__(self, root: str, subset: str, download: bool = True):
        super().__init__(root, download=download)

        def load_list(filename: str) -> List[str]:
            filepath = os.path.join(self._path, filename)
            with open(filepath, "r") as f:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in f]

        valid = set(load_list("validation_list.txt"))
        test = set(load_list("testing_list.txt"))

        all_files = [os.path.normpath(p) for p in self._walker]

        if subset == "validation":
            self._walker = [w for w in all_files if w in valid]
        elif subset == "testing":
            self._walker = [w for w in all_files if w in test]
        elif subset == "training":
            self._walker = [w for w in all_files if (w not in valid and w not in test)]
        else:
            raise ValueError("subset must be one of: training, validation, testing")

    def __getitem__(self, n: int):
        path = self._walker[n]
        waveform = load_audio(path)
        sample_rate = 16000
        label = label_from_path(path)

        base = os.path.splitext(os.path.basename(path))[0]
        parts = base.split("_")
        speaker_id = parts[0] if parts else ""
        utterance = parts[-1] if parts else ""

        return waveform, sample_rate, label, speaker_id, utterance

def label_from_path(path: str) -> str:
    return os.path.basename(os.path.dirname(path))

def filter_commands(dataset: SubsetSC, allowed: List[str] = COMMANDS, max_per_class: int | None = None):
    keep = []
    counts = {c: 0 for c in allowed}

    for p in dataset._walker:
        lab = label_from_path(p)
        if lab in counts:
            if max_per_class is None or counts[lab] < max_per_class:
                keep.append(p)
                counts[lab] += 1

    dataset._walker = keep
    return dataset

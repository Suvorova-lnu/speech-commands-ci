from __future__ import annotations
import numpy as np
import torch
import torchaudio
from scipy.io import wavfile
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

from .config import SAMPLE_RATE, CLIP_SAMPLES, N_MELS

_mel = MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=N_MELS)
_db = AmplitudeToDB()

def _np_audio_to_tensor(audio: np.ndarray) -> torch.Tensor:
    """Convert numpy audio array to torch float32 tensor (channels, samples) in [-1, 1]."""
    if audio.ndim == 1:
        audio = audio[:, None]  # (samples, 1)

    # now (samples, channels)
    if audio.dtype == np.int16:
        audio_f = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio_f = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.uint8:
        audio_f = (audio.astype(np.float32) - 128.0) / 128.0
    else:
        audio_f = audio.astype(np.float32)

    audio_f = np.clip(audio_f, -1.0, 1.0)
    audio_f = audio_f.T  # -> (channels, samples)
    return torch.from_numpy(audio_f)

def fix_length(waveform: torch.Tensor) -> torch.Tensor:
    # waveform: (channels, samples)
    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # mono
    if waveform.size(1) < CLIP_SAMPLES:
        pad = CLIP_SAMPLES - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    else:
        waveform = waveform[:, :CLIP_SAMPLES]
    return waveform

def waveform_to_logmelspec(waveform: torch.Tensor) -> torch.Tensor:
    waveform = fix_length(waveform)
    spec = _mel(waveform)          # (1, n_mels, time)
    spec_db = _db(spec)            # log scale
    return spec_db

def load_audio(path: str) -> torch.Tensor:
    """Load WAV using SciPy (avoids TorchCodec dependency on torchaudio.load in TorchAudio 2.9+)."""
    sr, audio = wavfile.read(path)
    wav = _np_audio_to_tensor(audio)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav

# Speech Commands (right / stop / up / go)

Проєкт: тренування KWS-моделі (PyTorch) + консольний інференс + веб-сторінка (Flask) + Docker.

## 1) Локальний запуск (Windows / Linux / macOS)

### Створити venv та встановити залежності
```bash
python -m venv .venv
# Windows PowerShell:
#   .\.venv\Scripts\Activate.ps1
# Якщо PowerShell блокує скрипти: запускай PowerShell як Admin і виконай:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt

> Примітка: у TorchAudio 2.9+ `torchaudio.load()` вимагає TorchCodec. У цьому проєкті WAV читаються через SciPy, тому TorchCodec не потрібен (стабільніше на Windows).

```

### Тренування
```bash
python -m src.train --device cpu --epochs 5 --max_per_class 5000
```

### Оцінка (запише models/metrics.json)
```bash
python -m src.evaluate --device cpu
```

### Консольний інференс
- Інференс з WAV файлу:
```bash
python -m src.infer_cli --audio path/to/sample.wav
```
- Інференс на випадковому тестовому прикладі певного класу:
```bash
python -m src.infer_cli --random_test_label right
```

### Веб локально (dev)
```bash
python -m src.webapp.app
# відкрий http://localhost:5000
```

## 2) Docker (веб + API)

> Важливо: перед `docker build` натренуй модель, щоб існувала папка `models/` з `model.pt` і `metrics.json`.

```bash
docker build -t speech-cmd .
docker run --rm -p 8000:5000 speech-cmd
# сторінка: http://localhost:8000
# метрики:  http://localhost:8000/metrics
```

API:
- POST /predict_text  JSON: {"label":"right"}
- POST /predict_audio multipart/form-data: file=<wav>
- GET  /metrics

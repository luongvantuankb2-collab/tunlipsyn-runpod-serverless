#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

python -m pip install -q "huggingface-hub==0.36.2"

if [ ! -f engines/LatentSync/checkpoints/latentsync_unet_v15.pt ] || [ ! -f engines/LatentSync/checkpoints/whisper/tiny.pt ]; then
  echo "[models] Downloading LatentSync checkpoints"
  bash download_checkpoints_ezycloud.sh
fi

if [ ! -f engines/LatentSync/checkpoints/sd-vae-ft-mse/config.json ]; then
  echo "[models] Downloading sd-vae-ft-mse"
  mkdir -p engines/LatentSync/checkpoints/sd-vae-ft-mse
  huggingface-cli download stabilityai/sd-vae-ft-mse \
    --local-dir engines/LatentSync/checkpoints/sd-vae-ft-mse
fi

if [ ! -f engines/LatentSync/checkpoints/auxiliary/models/buffalo_l/det_10g.onnx ]; then
  echo "[models] Downloading InsightFace buffalo_l"
  mkdir -p engines/LatentSync/checkpoints/auxiliary/models
  python - <<'PY'
from pathlib import Path
from shutil import copyfileobj
from urllib.request import Request, urlopen
import zipfile

root = Path("engines/LatentSync/checkpoints/auxiliary")
models_dir = root / "models"
models_dir.mkdir(parents=True, exist_ok=True)
zip_path = models_dir / "buffalo_l.zip"
url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urlopen(req, timeout=600) as response, zip_path.open("wb") as handle:
    copyfileobj(response, handle)
with zipfile.ZipFile(zip_path) as archive:
    archive.extractall(models_dir)
zip_path.unlink(missing_ok=True)
print("buffalo_l ready")
PY
fi

if [ ! -d voice_models/vieneu/VieNeu-TTS-v3-Turbo/onnx ]; then
  echo "[models] Downloading VieNeu-TTS-v3-Turbo"
  mkdir -p voice_models/vieneu
  huggingface-cli download pnnbao-ump/VieNeu-TTS-v3-Turbo \
    --revision d363ab07bbe11547528b3847386dc3d3273e5934 \
    --local-dir voice_models/vieneu/VieNeu-TTS-v3-Turbo
fi

test -f voice_models/vieneu/tao_giong_nhanh_vieneu.py
test -d voice_models/vieneu/VieNeu-TTS-v3-Turbo/onnx
test -f engines/LatentSync/checkpoints/latentsync_unet_v15.pt
test -f engines/LatentSync/checkpoints/whisper/tiny.pt
test -f engines/LatentSync/checkpoints/sd-vae-ft-mse/config.json
test -f engines/LatentSync/checkpoints/auxiliary/models/buffalo_l/det_10g.onnx

echo "[models] Ready"

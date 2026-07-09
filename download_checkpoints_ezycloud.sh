#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

python -m pip install -q "huggingface-hub==0.36.2"
mkdir -p engines/LatentSync/checkpoints

huggingface-cli download ByteDance/LatentSync-1.5 latentsync_unet.pt --local-dir engines/LatentSync/checkpoints/latentsync_v15_tmp
cp -f engines/LatentSync/checkpoints/latentsync_v15_tmp/latentsync_unet.pt engines/LatentSync/checkpoints/latentsync_unet_v15.pt
cp -f engines/LatentSync/checkpoints/latentsync_unet_v15.pt engines/LatentSync/checkpoints/latentsync_unet.pt

huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir engines/LatentSync/checkpoints/latentsync_v16_tmp
cp -f engines/LatentSync/checkpoints/latentsync_v16_tmp/latentsync_unet.pt engines/LatentSync/checkpoints/latentsync_unet_v16.pt

huggingface-cli download ByteDance/LatentSync-1.6 whisper/tiny.pt --local-dir engines/LatentSync/checkpoints

echo "LatentSync checkpoints are ready:"
find engines/LatentSync/checkpoints -maxdepth 3 -type f -printf "%p\n"

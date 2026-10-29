#!/usr/bin/env bash
set -euo pipefail
for model in resnet50_base resnet50_head resnet50_se resnet50_cbam; do
  python -m src.train --model "$model" --config configs/default.yaml --seed 42 --resume
  python -m src.evaluate --model "$model" --seed 42 --config configs/default.yaml
done
python -m src.compare --models resnet50_base resnet50_head resnet50_se resnet50_cbam

# FairFace gender classification with ResNet50

PyTorch project for binary gender classification (Male=1, Female=0) on FairFace. M0 is the ResNet50 baseline; M1 uses GeM and an MLP head; M2 adds SE blocks; M3 adds CBAM.

## Data

Use the FairFace **padding 0.25** images and the supplied train/validation label CSVs. Expected columns include `file`, `gender`, `race`, and `age`; supported alternate CSV filenames are searched automatically. Place files under `data/` locally, `/content/data/` on Colab, or attach them as a Kaggle Input. The program will stop with a clear error if labels/images cannot be found. It will not fabricate data or download a guessed dataset.

```bash
python scripts/prepare_splits.py --data_dir data/fairface
```

The original train partition is split into train/validation (90/10, stratified by gender and race); the original FairFace validation partition is held out as test. Generated split lists go under `splits/`.

## Run locally

Install PyTorch and torchvision matching your machine, then install the remaining dependencies with `pip install -r requirements.txt`.

```bash
python scripts/prepare_splits.py --data_dir data/fairface
python -m src.train --model resnet50_base --config configs/default.yaml --seed 42 --resume
python -m src.evaluate --model resnet50_base --seed 42
python -m src.compare --models resnet50_base resnet50_head resnet50_se resnet50_cbam
```

`scripts/run_all.sh` runs the four models in sequence. Add `--epochs N` to each training command consistently when running a shorter experiment. `--debug` runs two epochs at batch size 8. Checkpoints are saved every epoch and `--resume` restores optimizer, scaler, history, and RNG state. Completed runs with a `DONE` marker are skipped.

## Colab

Upload/mount your FairFace data at `/content/data`, or edit `data_source` and provide your own source details. For persistence, mount Google Drive and set `output_dir` to `/content/drive/MyDrive/fairface_gender/outputs`. ImageNet pretrained weights require internet unless `pretrained_path` points to a compatible ResNet50 state dictionary.

## Kaggle

Enable GPU and attach FairFace through **Add Input**. The project searches `/kaggle/input` and writes artifacts to `/kaggle/working/outputs`; it never copies the image dataset into working storage. Download checkpoints before the session ends or save them as a Kaggle Dataset. If internet is disabled, supply ImageNet weights with `pretrained_path`.

The README intentionally leaves Kaggle dataset slug and download URL unspecified; supply values from your own chosen source rather than relying on an assumed dataset mirror.

## Outputs and limits

Each run writes `last.pt`, `best.pt`, epoch history, config, test metrics, predictions, and race/age fairness CSVs. Comparison writes summary tables, an ROC plot, and exact McNemar tests. FLOPs and Grad-CAM visualization export are not automated in this initial implementation; `src/gradcam.py` provides the M0 layer4 Grad-CAM calculation helper. Results are generated only after training on actual FairFace data.

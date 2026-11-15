# FairFace ResNet50 gender experiments

Python project for the Kaggle **GPU T4 x2** runtime. It trains FairFace binary gender classifiers (Male=0, Female=1) using the official validation split as test and a stratified 10% development split from official training data.

## Environment

Python 3.10+, PyTorch with CUDA, torchvision, and the packages in `requirements.txt`. Kaggle supplies the CUDA-enabled PyTorch/torchvision build; the notebook installs the remaining dependencies from this file so pip does not replace Kaggle's GPU build with a CPU wheel.

## Prepare and train

```bash
cd fairface_project
python -m pip install -r requirements.txt
python -m fairface.data --root /kaggle/working --margin 1.25 --size 224
NCCL_IB_DISABLE=1 OMP_NUM_THREADS=2 torchrun --standalone --nproc_per_node=2 -m fairface.train \
  --model m1 --cache /kaggle/working/fairface_1.25_cache \
  --out /kaggle/working/results --bs 128
```

The Kaggle notebooks in `notebooks/` clone this GitHub repository at the selected branch, install `requirements.txt`, and run M1–M2 or M3–M4 from data download through training and result reporting. Set the Kaggle accelerator to **GPU T4 x2** and enable Internet. Do not attach a Kaggle Input dataset. The notebook config cell contains the repository URL and branch and can be changed if you fork this repository.

## Models

- M1: ImageNet V1 ResNet50, linear binary head, basic augmentation, up to 15 epochs.
- M2: M1 with two-stage frozen-head/full fine-tuning, discriminative layer learning rates, stronger augmentation, label smoothing.
- M3: M2 with CBAM stage gates, GeM, MLP head, EMA.
- M4: M3 with ImageNet V2 weights, progressive 128/176/224 resizing, mixup, and horizontal-flip test-time augmentation.

All models use a maximum of 15 epochs and early stopping on dev loss (`patience=5`, `min_delta=1e-3`, starting at epoch 8). The best checkpoint is selected by minimum dev loss. M2–M4 use 2 head-only epochs followed by 13 fine-tuning epochs; M4's fine-tuning resolution schedule is 5 epochs at 128, 4 at 176, and 4 at 224 pixels.

Each run writes `{model}_best.pt`, `{model}_last.pt`, `{model}_log.csv`, `{model}_metrics.json`, `{model}_test_logits.npy`, and test labels to the output directory. `last.pt` supports restart at an epoch boundary. Checkpoints and result files should be downloaded from Kaggle's output panel before the session ends.

from __future__ import annotations
import argparse, json, os, shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def prepare(root: str, margin: str = "1.25", size: int = 224, seed: int = 42, workers: int = 4):
    root = Path(root); cache = root / f"fairface_{margin}_cache"; cache.mkdir(parents=True, exist_ok=True)
    hf_cache = root / "hf_cache"
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("Install `datasets` before preparing FairFace") from e
    print(f"Loading Hugging Face FairFace config={margin}", flush=True)
    ds = load_dataset("HuggingFaceM4/FairFace", margin, cache_dir=str(hf_cache))
    tr, va = ds["train"], ds["validation"]
    print(ds); print("Features:", tr.features); print("First image size:", tr[0]["image"].size)
    gender_names = tr.features["gender"].names
    race_names = tr.features["race"].names
    assert "Male" in gender_names and "Female" in gender_names, gender_names
    male, female = gender_names.index("Male"), gender_names.index("Female")
    y_gender = np.asarray(tr["gender"], dtype=np.int64)
    y_race = np.asarray(tr["race"], dtype=np.int64)
    # Stratification preserves every race/gender combination where present.
    strata = np.asarray([f"{r}:{g}" for r, g in zip(y_race, y_gender)])
    all_idx = np.arange(len(tr))
    train_idx, dev_idx = train_test_split(all_idx, test_size=.10, random_state=seed, stratify=strata)
    np.save(cache / "trainmain_idx.npy", train_idx); np.save(cache / "dev_idx.npy", dev_idx)
    meta = {"margin": margin, "size": size, "seed": seed, "gender_names": gender_names,
            "race_names": race_names, "male_index": male, "female_index": female,
            "train_count": len(tr), "test_count": len(va)}
    (cache / "meta.json").write_text(json.dumps(meta, indent=2))

    def build(split, prefix):
        n = len(split); X = np.lib.format.open_memmap(cache / f"{prefix}_x.npy", mode="w+", dtype=np.uint8, shape=(n,size,size,3))
        gy = np.lib.format.open_memmap(cache / f"{prefix}_gender.npy", mode="w+", dtype=np.float32, shape=(n,))
        ry = np.lib.format.open_memmap(cache / f"{prefix}_race.npy", mode="w+", dtype=np.int8, shape=(n,))
        def decode(im):
            im = im.convert("RGB")
            if im.size != (size,size): im = im.resize((size,size), Image.Resampling.BILINEAR)
            return np.asarray(im, dtype=np.uint8)
        chunk = 1024
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0,n,chunk):
                end = min(start+chunk,n); rows = split[start:end]
                X[start:end] = np.stack(list(pool.map(decode, rows["image"])))
                gy[start:end] = np.asarray(rows["gender"],dtype=np.int64) == female
                ry[start:end] = rows["race"]
                print(f"{prefix}: {end}/{n}", flush=True)
        X.flush(); gy.flush(); ry.flush()
    build(tr, "train"); build(va, "test")
    shutil.rmtree(hf_cache, ignore_errors=True)
    print("Cache bytes:", sum(p.stat().st_size for p in cache.glob("*.npy")))
    print("Prepared at", cache)
    return cache

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--root",required=True); ap.add_argument("--margin",default="1.25"); ap.add_argument("--size",type=int,default=224)
    a=ap.parse_args(); prepare(a.root,a.margin,a.size)

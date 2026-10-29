"""Runtime and FairFace path discovery helpers."""
from __future__ import annotations
import importlib.util, os
from pathlib import Path

def detect_env() -> str:
    if os.getenv("KAGGLE_KERNEL_RUN_TYPE") or Path('/kaggle/input').exists(): return 'kaggle'
    if os.getenv('COLAB_GPU') or importlib.util.find_spec('google.colab'): return 'colab'
    return 'local'

def find_label_files(root: str | Path) -> tuple[Path | None, Path | None]:
    root=Path(root)
    if not root.exists(): return None,None
    files=list(root.rglob('*.csv'))
    def pick(kind: str):
        terms=('train_labels','fairface_label_train') if kind=='train' else ('val_labels','fairface_label_val')
        matches=[p for p in files if any(t in p.name.lower() for t in terms)]
        return matches[0] if matches else None
    return pick('train'),pick('val')

def resolve_paths(cfg: dict) -> dict:
    env=detect_env()
    defaults={'kaggle':('/kaggle/input','/kaggle/working/outputs'), 'colab':('/content/data','/content/drive/MyDrive/fairface_gender/outputs'), 'local':('./data','./outputs')}
    data=Path(cfg.get('data_dir') or defaults[env][0]).expanduser()
    out=Path(cfg.get('output_dir') or defaults[env][1]).expanduser()
    train,val=find_label_files(data)
    # Kaggle datasets are mounted under input; search there if no explicit dataset subdirectory.
    if (train is None or val is None) and env=='kaggle': train,val=find_label_files('/kaggle/input')
    if train and val:
        data=train.parent
        print(f'Found FairFace labels: train={train}, val={val}')
    out.mkdir(parents=True,exist_ok=True)
    splits=Path(cfg.get('splits_dir') or (out.parent/'splits'))
    return {'env':env,'data_dir':data,'output_dir':out,'splits_dir':splits,'train_csv':train,'test_csv':val}

def find_image_root(csv_path: Path, frame) -> Path:
    """Find directory that resolves the first few paths in the CSV."""
    candidates=[csv_path.parent, csv_path.parent.parent]
    for col in ('file','image','filename'):
        if col in frame:
            sample=[str(x) for x in frame[col].dropna().head(5)]
            for base in candidates:
                if sample and sum((base/p).exists() for p in sample)>=min(3,len(sample)): return base
    return csv_path.parent

import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.env import resolve_paths,find_image_root
from src.dataset import normalize_columns

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data_dir'); p.add_argument('--out'); a=p.parse_args()
    paths=resolve_paths({'data_dir':a.data_dir,'output_dir':'./outputs'})
    train_csv,test_csv=paths['train_csv'],paths['test_csv']
    if not train_csv or not test_csv: raise SystemExit('FairFace CSVs not found. Supply --data_dir containing train_labels.csv and val_labels.csv (or fairface_label_train/val.csv).')
    train=normalize_columns(pd.read_csv(train_csv)); test=normalize_columns(pd.read_csv(test_csv))
    print('Train/test shapes:',train.shape,test.shape); print('Gender:',train.gender.value_counts(dropna=False).to_dict()); print('Race:',train.race.value_counts(dropna=False).to_dict())
    root=find_image_root(train_csv,train); print('Image root:',root)
    print('Sample files:',[(str(x), (root/str(x)).exists()) for x in train.file.head(5)])
    if not all((root/str(x)).exists() for x in train.file.head(5)): raise SystemExit('Image paths do not resolve. Set data_dir to the FairFace directory containing labels and images.')
    tr,va=train_test_split(train,test_size=.1,random_state=42,stratify=train.gender.astype(str)+'|'+train.race.astype(str))
    out=Path(a.out or paths['splits_dir']); out.mkdir(parents=True,exist_ok=True)
    tr.to_csv(out/'train.csv',index=False); va.to_csv(out/'val.csv',index=False); test.to_csv(out/'test.csv',index=False)
    sets=[set(x.file.astype(str)) for x in (tr,va,test)]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i)): raise RuntimeError('File leakage across splits')
    for name,df in [('train',tr),('val',va),('test',test)]: print(name,'gender',df.gender.value_counts().to_dict(),'race',df.race.value_counts().to_dict())
if __name__=='__main__': main()

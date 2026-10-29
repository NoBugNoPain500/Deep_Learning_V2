from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

MEAN=(.485,.456,.406); STD=(.229,.224,.225)
def normalize_columns(df):
    mapping={str(c).strip().lower():c for c in df.columns}
    aliases={'file':('file','image','filename','img'),'gender':('gender','sex'),'race':('race','ethnicity'),'age':('age','age_group')}
    rename={}
    for target,names in aliases.items():
        found=next((mapping[n] for n in names if n in mapping),None)
        if found is None and target in ('file','gender'): raise ValueError(f'Missing required {target} column; found {list(df.columns)}')
        if found: rename[found]=target
    return df.rename(columns=rename)

class FairFaceDataset(Dataset):
    def __init__(self,frame,image_root,train=False):
        self.frame=normalize_columns(frame).reset_index(drop=True); self.root=Path(image_root)
        self.transform=transforms.Compose(([transforms.RandomResizedCrop(224,scale=(.8,1.)),transforms.RandomHorizontalFlip(.5),transforms.RandomRotation(15),transforms.ColorJitter(brightness=.15,hue=.04)] if train else [transforms.Resize((224,224))])+[transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
        self.labels={'female':0,'male':1}
    def __len__(self): return len(self.frame)
    def __getitem__(self,index):
        row=self.frame.iloc[index]; path=self.root/str(row.file)
        with Image.open(path) as im: image=self.transform(im.convert('RGB'))
        gender=str(row.gender).strip().lower()
        if gender not in self.labels: raise ValueError(f'Unexpected gender label: {row.gender}')
        return image,self.labels[gender],{k:row.get(k,'Unknown') for k in ('race','age')}

def make_loader(ds,batch_size,workers=0,shuffle=False):
    return DataLoader(ds,batch_size=batch_size,shuffle=shuffle,num_workers=workers,pin_memory=True,persistent_workers=workers>0)

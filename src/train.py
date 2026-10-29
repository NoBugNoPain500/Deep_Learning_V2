from __future__ import annotations
import argparse,csv,time,yaml,os
from pathlib import Path
import pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from .env import resolve_paths,find_image_root
from .dataset import FairFaceDataset
from .models import build_model
from .utils import set_seed,rng_state,restore_rng,count_parameters

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--seed',type=int,default=42); p.add_argument('--resume',action='store_true'); p.add_argument('--data_dir'); p.add_argument('--output_dir'); p.add_argument('--epochs',type=int); p.add_argument('--debug',action='store_true'); a=p.parse_args()
    cfg=yaml.safe_load(open(a.config)); cfg['data_dir']=a.data_dir or cfg.get('data_dir'); cfg['output_dir']=a.output_dir or cfg.get('output_dir'); cfg['seed']=a.seed
    if a.epochs: cfg['epochs']=a.epochs
    paths=resolve_paths(cfg); split=paths['splits_dir']; trpath=split/'train.csv'; vapath=split/'val.csv'
    if not trpath.exists() or not vapath.exists(): raise SystemExit('Splits missing. Run python scripts/prepare_splits.py first.')
    run=paths['output_dir']/(('debug/'+a.model) if a.debug else a.model)/str(a.seed); run.mkdir(parents=True,exist_ok=True)
    if (run/'DONE').exists(): print(f'{a.model} seed={a.seed} already DONE; skipping'); return
    set_seed(a.seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train=pd.read_csv(trpath); val=pd.read_csv(vapath); root=find_image_root(paths['train_csv'] or Path(cfg['data_dir'] or './data'),train)
    td=FairFaceDataset(train,root,True); vd=FairFaceDataset(val,root,False); workers=cfg.get('num_workers') or min(4,os.cpu_count() or 1)
    bs=8 if a.debug else int(cfg['batch_size']); model=build_model(a.model,True,cfg.get('pretrained_path')).to(device)
    if cfg.get('multi_gpu') and torch.cuda.device_count()>1: model=nn.DataParallel(model)
    print(f'Parameters: {count_parameters(model):,}; device={device}; batch={bs}; workers={workers}')
    head=(model.module if isinstance(model,nn.DataParallel) else model).fc
    head_ids={id(p) for p in head.parameters()}; backbone=[p for p in model.parameters() if id(p) not in head_ids]
    opt=torch.optim.AdamW([{'params':backbone,'lr':cfg['lr_backbone']},{'params':head.parameters(),'lr':cfg['lr_head']}],weight_decay=cfg['weight_decay'])
    total_epochs=2 if a.debug or cfg.get('debug') else cfg['epochs']; scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and cfg.get('amp',True)); start=0; best=0.; stale=0; history=[]
    last=run/'last.pt'
    if a.resume and last.exists():
        ck=torch.load(last,map_location=device,weights_only=False); model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler']); start=ck['epoch']+1; best=ck['best']; stale=ck['stale']; history=ck['history']; restore_rng(ck['rng'])
    with open(run/'config.yaml','w') as f: yaml.safe_dump(cfg,f)
    lossfn=nn.BCEWithLogitsLoss(); since=time.time()
    for epoch in range(start,total_epochs):
        # Linear warmup followed by cosine schedule, same epoch schedule for all models.
        scale=(epoch+1)/max(1,cfg['warmup_epochs']) if epoch<cfg['warmup_epochs'] else .5*(1+__import__('math').cos(__import__('math').pi*(epoch-cfg['warmup_epochs'])/max(1,total_epochs-cfg['warmup_epochs'])))
        for group,base in zip(opt.param_groups,(cfg['lr_backbone'],cfg['lr_head'])): group['lr']=base*scale
        model.train(); trloss=correct=n=0; loader=DataLoader(td,batch_size=bs,shuffle=True,num_workers=workers,pin_memory=True,persistent_workers=workers>0)
        opt.zero_grad(set_to_none=True); accum=max(1,round(cfg['effective_batch_size']/bs))
        for i,(x,y,_) in enumerate(loader):
            x=x.to(device); y=y.float().to(device).view(-1,1)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda' and cfg.get('amp',True)): logits=model(x); loss=lossfn(logits,y)/accum
            scaler.scale(loss).backward()
            if (i+1)%accum==0 or i+1==len(loader): scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip']); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            trloss+=loss.item()*accum*len(y); correct+=((logits>=0)==(y>=.5)).sum().item(); n+=len(y)
        model.eval(); vloss=vc=vn=0
        with torch.no_grad():
            for x,y,_ in DataLoader(vd,batch_size=bs,num_workers=workers,pin_memory=True):
                x=x.to(device); y=y.float().to(device).view(-1,1); z=model(x); vloss+=lossfn(z,y).item()*len(y); vc+=((z>=0)==(y>=.5)).sum().item(); vn+=len(y)
        row={'epoch':epoch+1,'train_loss':trloss/n,'train_acc':correct/n,'val_loss':vloss/vn,'val_acc':vc/vn,'epoch_seconds':time.time()-since}; history.append(row); since=time.time(); print(row)
        pd.DataFrame(history).to_csv(run/'history.csv',index=False)
        state={'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'best':best,'stale':stale,'history':history,'rng':rng_state()}; torch.save(state,last)
        if vc/vn>best: best=vc/vn; stale=0; state['best']=best; torch.save(state,run/'best.pt')
        else: stale+=1
        if stale>=cfg['patience']: break
    (run/'DONE').touch()
if __name__=='__main__': main()

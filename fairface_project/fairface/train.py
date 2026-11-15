from __future__ import annotations
import argparse, csv, datetime, json, math, os, random, time
from pathlib import Path
import numpy as np, torch, torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_fscore_support
from .models import ResNet50Gender, MemmapLoader, gpu_prep, EMA

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True,choices=['m1','m2','m3','m4']); ap.add_argument('--cache',required=True); ap.add_argument('--out',required=True); ap.add_argument('--epochs',type=int,default=0); ap.add_argument('--bs',type=int,default=96); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--sanity',action='store_true')
 a=ap.parse_args(); rank=int(os.getenv('RANK','0')); world=int(os.getenv('WORLD_SIZE','1')); local=int(os.getenv('LOCAL_RANK','0')); torch.cuda.set_device(local)
 if world>1: dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=30))
 random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); torch.backends.cudnn.benchmark=True
 cache=Path(a.cache); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
 X=np.load(cache/'train_x.npy',mmap_mode='r'); y=np.load(cache/'train_gender.npy',mmap_mode='r'); races=np.load(cache/'train_race.npy',mmap_mode='r'); testx=np.load(cache/'test_x.npy',mmap_mode='r'); testy=np.load(cache/'test_gender.npy',mmap_mode='r'); testr=np.load(cache/'test_race.npy',mmap_mode='r')
 train_idx=np.load(cache/'trainmain_idx.npy'); dev_idx=np.load(cache/'dev_idx.npy'); bs=a.bs
 train=MemmapLoader(X,y,bs,train_idx,True,True,a.seed,rank,world); dev=MemmapLoader(X,y,bs,dev_idx); test=MemmapLoader(testx,testy,bs)
 if world>1 and rank!=0: dist.barrier()
 model=ResNet50Gender(a.model).cuda().to(memory_format=torch.channels_last)
 if world>1 and rank==0: dist.barrier()
 if a.sanity:
  xb,yb=next(iter(MemmapLoader(X,y,16,train_idx))); aug=gpu_prep(xb,224,True,True); outb=model(aug); assert outb.shape==(16,)
  if rank==0:
   from PIL import Image, ImageDraw
   imgs=(aug.float()*aug.new_tensor([.229,.224,.225]).view(1,3,1,1)+aug.new_tensor([.485,.456,.406]).view(1,3,1,1)).clamp(0,1).mul(255).byte().permute(0,2,3,1).cpu().numpy()
   sheet=Image.new('RGB',(4*224,4*244),'white'); draw=ImageDraw.Draw(sheet)
   for i,img in enumerate(imgs):
    x0=(i%4)*224; y0=(i//4)*244; sheet.paste(Image.fromarray(img),(x0,y0)); draw.text((x0+4,y0+225),f'label={int(yb[i].item())}',fill='black')
   sheet.save(out/'aug_check.png'); print('Saved',out/'aug_check.png')
  print('forward sanity passed',flush=True)
  if world>1: dist.destroy_process_group()
  return
 total=a.epochs or 15; two=a.model!='m1'; ema=EMA(model) if a.model in ('m3','m4') else None
 best=float('inf'); best_loss=best; best_acc=0.; rows=[]; start_ep=0; patience_count=0; previous_resolution=None; stop_training=False; ckpt=out/f'{a.model}_last.pt'; bestpath=out/f'{a.model}_best.pt'; resume_state=None
 if ckpt.exists():
  resume_state=torch.load(ckpt,map_location=f'cuda:{local}',weights_only=False); model.load_state_dict(resume_state['model']); start_ep=resume_state['epoch']+1; best=resume_state.get('best_loss',resume_state.get('best',float('inf'))); best_loss=resume_state.get('best_loss',best); best_acc=resume_state.get('best_accuracy',0.); patience_count=resume_state.get('patience_count',0); previous_resolution=resume_state.get('resolution'); rows=resume_state['rows'];
  if ema and resume_state.get('ema'): ema.model.load_state_dict(resume_state['ema'])
 opt=sched=scaler=None; active_stage=None
 for ep in range(start_ep,total):
  stage1=two and ep<2
  for p in model.parameters(): p.requires_grad_(True)
  if stage1:
   for p in model.stem.parameters(): p.requires_grad_(False)
   for p in model.layers.parameters(): p.requires_grad_(False)
  phase=stage1 if two else False
  if active_stage != phase:
   if world>1 and 'module' in locals(): del module
   if phase:
    groups=[{'params':[p for p in model.parameters() if p.requires_grad], 'lr':1e-3}]
   elif two:
    specs=[('stem',.1),('layers.0',.1),('layers.1',.25),('layers.2',.5),('layers.3',1.),('gates',2.5),('pool',2.5),('head',2.5)]
    grouped={}
    for n,p in model.named_parameters():
     if not p.requires_grad: continue
     mult=next((v for key,v in specs if n==key or n.startswith(key+'.')),1.)
     decay=p.ndim>1; key=(mult,decay); grouped.setdefault(key,[]).append(p)
    groups=[{'params':ps,'lr':4e-4*m,'weight_decay':.02 if d else 0.} for (m,d),ps in grouped.items()]
   else: groups=[{'params':model.parameters(),'lr':2e-4,'weight_decay':.01}]
   opt=torch.optim.AdamW(groups,lr=1e-3 if phase else 4e-4,fused=True)
   phase_epochs=(2 if phase else (total-2 if two else total)); nsteps=max(1,phase_epochs*len(train)); warm=len(train) if two and not phase else 0
   def lr_factor(step):
    warm_factor=min(1.,(step+1)/max(1,warm)) if warm else 1.
    progress=max(0,min(1,(step-warm)/max(1,nsteps-warm)))
    return warm_factor*.5*(1+math.cos(math.pi*progress)) if not phase else 1.
   sched=torch.optim.lr_scheduler.LambdaLR(opt,lr_factor); scaler=torch.amp.GradScaler('cuda')
   if resume_state is not None and resume_state.get('stage1')==phase and resume_state.get('optimizer') is not None:
    opt.load_state_dict(resume_state['optimizer']); sched.load_state_dict(resume_state['scheduler']); scaler.load_state_dict(resume_state['scaler'])
   resume_state=None; active_stage=phase
  module=DDP(model,device_ids=[local],gradient_as_bucket_view=True) if world>1 else model
  module.train()
  if stage1: model.stem.eval(); model.layers.eval()
  train.set_epoch(ep); tic=time.time(); res=(128 if a.model=='m4' and ep<7 else (176 if a.model=='m4' and ep<11 else 224)); standard=two
  if a.model=='m4' and previous_resolution is not None and res != previous_resolution: patience_count=0
  loss_sum=torch.zeros((),device='cuda'); count=torch.zeros((),device='cuda')
  for xb,yb in train:
   x=gpu_prep(xb,res,True,standard); target=yb
   if a.model=='m4' and ep<total-3 and np.random.rand()<.5:
    lam=float(np.random.beta(.2,.2)); perm=torch.randperm(len(x),device=x.device); x=lam*x+(1-lam)*x[perm]; target=lam*yb+(1-lam)*yb[perm]
   if two: target=target*.95+.025
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16): logits=module(x)
   loss=F.binary_cross_entropy_with_logits(logits.float(),target); scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(),1.); scaler.step(opt); scaler.update(); sched.step()
   if ema and not stage1: ema.update(model)
   loss_sum+=loss.detach()*len(yb); count+=len(yb)
  if world>1: dist.all_reduce(loss_sum); dist.all_reduce(count)
  train_loss=(loss_sum/count).item(); elapsed=time.time()-tic
  if rank==0:
   evaluator=ema.model if ema and ep>=2 else model; evaluator.eval(); logits_all=[]; labels=[]
   with torch.no_grad():
    for xb,yb in dev:
     x=gpu_prep(xb,224,False)
     with torch.autocast('cuda',dtype=torch.float16): z=evaluator(x)
     logits_all.append(z.float().cpu().numpy()); labels.append(yb.cpu().numpy())
   dz=np.concatenate(logits_all); dy=np.concatenate(labels); pred=dz>=0; acc=float((pred==dy).mean()); dl=float(np.logaddexp(0,dz).mean()-np.mean(dy*dz))
   row={'epoch':ep+1,'train_loss':train_loss,'dev_loss':dl,'dev_accuracy':acc,'seconds':elapsed,'resolution':res}; rows.append(row); print(json.dumps(row),flush=True)
   checkpoint_improved=dl < best_loss
   patience_improved=dl < best_loss-1e-3
   if checkpoint_improved: best_loss=dl; best=dl; best_acc=acc
   if patience_improved: patience_count=0
   elif ep+1>=8: patience_count+=1
   state={'model':model.state_dict(),'epoch':ep,'best':best,'best_loss':best_loss,'best_accuracy':best_acc,'patience_count':patience_count,'resolution':res,'rows':rows,'ema':ema.model.state_dict() if ema else None,'stage1':stage1 if two else False,'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'scaler':scaler.state_dict()}; torch.save(state,ckpt)
   if checkpoint_improved: torch.save(state,bestpath)
   stop_training=ep+1>=8 and patience_count>=5
   previous_resolution=res
  if world>1:
   flag=torch.tensor([int(stop_training)],device='cuda'); dist.broadcast(flag,src=0); stop_training=bool(flag.item())
  if world>1: dist.barrier()
  if world>1: del module
  if stop_training: break
 if rank==0:
  state=torch.load(bestpath,map_location=f'cuda:{local}',weights_only=False); model.load_state_dict(state['model']); evaluator=model
  if ema and state.get('ema'): ema.model.load_state_dict(state['ema']); evaluator=ema.model
  evaluator.eval(); logits=[]; labels=[]; rr=[]
  with torch.no_grad():
   for xb,yb in test:
    x=gpu_prep(xb,224,False)
    with torch.autocast('cuda',dtype=torch.float16): z=evaluator(x)
    if a.model=='m4':
     with torch.autocast('cuda',dtype=torch.float16): z=(z+evaluator(gpu_prep(xb.flip(2),224,False)))/2
    logits.append(z.float().cpu().numpy()); labels.append(yb.cpu().numpy())
  z=np.concatenate(logits); yt=np.concatenate(labels).astype(int); yp=(z>=0).astype(int); race=np.asarray(testr)
  def infer_speed(batch,repeats=30):
   dummy=torch.zeros(batch,3,224,224,device='cuda').contiguous(memory_format=torch.channels_last)
   for _ in range(5):
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16): evaluator(dummy)
   torch.cuda.synchronize(); start=time.perf_counter()
   with torch.no_grad():
    for _ in range(repeats):
     with torch.autocast('cuda',dtype=torch.float16): evaluator(dummy)
   torch.cuda.synchronize(); return (time.perf_counter()-start)*1000/repeats
  latency_ms=infer_speed(1); latency64_ms=infer_speed(64)
  metric={'model':a.model,'class_order':['Male','Female'],'parameter_count':sum(p.numel() for p in evaluator.parameters()),'train_minutes':sum(r['seconds'] for r in rows)/60,'train_images_per_second':len(train_idx)*len(rows)/max(1e-6,sum(r['seconds'] for r in rows)),'peak_gpu_memory_gb':torch.cuda.max_memory_allocated()/1024**3,'inference_latency_ms_batch1':latency_ms,'inference_images_per_second_batch64':64*1000/latency64_ms,'test_accuracy':float(accuracy_score(yt,yp)),'f1_female':float(f1_score(yt,yp)),'roc_auc':float(roc_auc_score(yt,z)),'confusion_matrix':confusion_matrix(yt,yp).tolist(),'precision_recall_by_class':precision_recall_fscore_support(yt,yp,labels=[0,1],zero_division=0)[0:2]}
  metric['precision_recall_by_class']=[np.asarray(v).tolist() for v in metric['precision_recall_by_class']]
  metric['race_accuracy']={str(int(r)):float(accuracy_score(yt[race==r],yp[race==r])) for r in np.unique(race)}; vals=list(metric['race_accuracy'].values()); metric['race_accuracy_gap']=max(vals)-min(vals)
  (out/f'{a.model}_metrics.json').write_text(json.dumps(metric,indent=2)); np.save(out/f'{a.model}_test_logits.npy',z); np.save(out/f'{a.model}_test_labels.npy',yt)
  with open(out/f'{a.model}_log.csv','w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
  print(json.dumps(metric,indent=2),flush=True)
 if world>1: dist.barrier(); dist.destroy_process_group()
if __name__=='__main__': main()

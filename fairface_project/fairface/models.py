from __future__ import annotations
import copy, math, queue, threading
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision

class CBAMGate(nn.Module):
    def __init__(self,c,r=16):
        super().__init__(); self.mlp=nn.Sequential(nn.Conv2d(c,c//r,1),nn.ReLU(inplace=True),nn.Conv2d(c//r,c,1)); self.spatial=nn.Conv2d(2,1,7,padding=3); self.alpha=nn.Parameter(torch.zeros(1))
    def forward(self,x):
        ca=torch.sigmoid(self.mlp(x.mean((2,3),keepdim=True))+self.mlp(x.amax((2,3),keepdim=True))); y=x*ca
        sa=torch.sigmoid(self.spatial(torch.cat((y.mean(1,keepdim=True),y.amax(1,keepdim=True)),1))); y=y*sa
        return x+self.alpha*(y-x)
class GeM(nn.Module):
    def __init__(self): super().__init__(); self.p=nn.Parameter(torch.ones(1)*3); self.eps=1e-6
    def forward(self,x): return x.float().clamp(min=self.eps).pow(self.p).mean((2,3)).pow(1/self.p)
class ResNet50Gender(nn.Module):
    def __init__(self,model_id="m1"):
        super().__init__(); advanced=model_id in ("m3","m4"); weights=None if model_id=="m0" else (torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if model_id=="m4" else torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        b=torchvision.models.resnet50(weights=weights)
        self.stem=nn.Sequential(b.conv1,b.bn1,b.relu,b.maxpool); self.layers=nn.ModuleList([b.layer1,b.layer2,b.layer3,b.layer4])
        self.gates=nn.ModuleList([CBAMGate(c) for c in (256,512,1024,2048)]) if advanced else None
        self.pool=GeM() if advanced else nn.AdaptiveAvgPool2d(1)
        self.head=nn.Sequential(nn.Dropout(.3),nn.Linear(2048,512),nn.GELU(),nn.Dropout(.3),nn.Linear(512,1)) if advanced else nn.Linear(2048,1)
    def forward(self,x):
        x=self.stem(x)
        for i,l in enumerate(self.layers):
            x=l(x)
            if self.gates is not None: x=self.gates[i](x)
        return self.head(self.pool(x).flatten(1)).squeeze(1)

class MemmapLoader:
    def __init__(self,x,y,bs,indices=None,shuffle=False,drop_last=False,seed=42,rank=0,world=1):
        self.x=x; self.y=y; self.bs=bs; self.indices=np.arange(len(y)) if indices is None else np.asarray(indices); self.shuffle=shuffle; self.drop_last=drop_last; self.seed=seed; self.rank=rank; self.world=world; self.epoch=0
    def set_epoch(self,e): self.epoch=e
    def __len__(self):
        n=(len(self.indices)//self.world) if self.world>1 else len(self.indices)
        return n//self.bs if self.drop_last else math.ceil(n/self.bs)
    def __iter__(self):
        order=self.indices
        if self.shuffle: order=np.random.default_rng(self.seed+self.epoch).permutation(order)
        if self.world>1: order=order[:(len(order)//self.world)*self.world][self.rank::self.world]
        q=queue.Queue(4)
        def worker():
            for i in range(0,len(order),self.bs):
                ids=order[i:i+self.bs]
                if self.drop_last and len(ids)<self.bs: break
                ids=np.sort(ids); q.put((torch.from_numpy(np.array(self.x[ids])).pin_memory(),torch.from_numpy(np.array(self.y[ids])).float()))
            q.put(None)
        threading.Thread(target=worker,daemon=True).start()
        while True:
            item=q.get()
            if item is None: break
            xb,yb=item; yield xb.cuda(non_blocking=True),yb.cuda(non_blocking=True)

def gpu_prep(xb,size,training,standard=False):
    x=xb.permute(0,3,1,2).float().div_(255); b=x.shape[0]
    if training:
        scale=(torch.empty(b,device=x.device).uniform_(.7 if standard else .8,1.).sqrt())
        angle=torch.empty(b,device=x.device).uniform_(-10 if standard else 0,10 if standard else 0)*math.pi/180
        flip=torch.where(torch.rand(b,device=x.device)<.5,-1.,1.); tx=(torch.rand(b,device=x.device)*2-1)*(1-scale); ty=(torch.rand(b,device=x.device)*2-1)*(1-scale)
        c,s=torch.cos(angle),torch.sin(angle)
        theta=torch.stack((torch.stack((scale*c*flip,-scale*s,tx),1),torch.stack((scale*s*flip,scale*c,ty),1)),1)
        grid=F.affine_grid(theta,(b,3,size,size),align_corners=False); x=F.grid_sample(x,grid,mode="bilinear",padding_mode="reflection",align_corners=False)
        if standard:
            k=.2; br=1+(torch.rand(b,1,1,1,device=x.device)*2-1)*k; ct=1+(torch.rand(b,1,1,1,device=x.device)*2-1)*k; mean=x.mean((1,2,3),keepdim=True); x=((x-mean)*ct+mean).mul(br).clamp_(0,1)
    elif x.shape[-1]!=size: x=F.interpolate(x,(size,size),mode="bilinear",antialias=True,align_corners=False)
    mean=x.new_tensor([.485,.456,.406]).view(1,3,1,1); std=x.new_tensor([.229,.224,.225]).view(1,3,1,1)
    return ((x-mean)/std).contiguous(memory_format=torch.channels_last)

class EMA:
    def __init__(self,model,decay=.998):
        self.model=copy.deepcopy(model).eval(); self.decay=decay
        for p in self.model.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self,model):
        for e,p in zip(self.model.parameters(),model.parameters()): e.lerp_(p.detach(),1-self.decay)
        for e,b in zip(self.model.buffers(),model.buffers()): e.copy_(b)

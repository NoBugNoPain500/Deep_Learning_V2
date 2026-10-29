import torch
from torch import nn
from .base import resnet

class GeM(nn.Module):
    def __init__(self,p=3.,eps=1e-6): super().__init__(); self.p=nn.Parameter(torch.tensor(float(p))); self.eps=eps
    def forward(self,x): return x.clamp(min=self.eps).pow(self.p).mean((-2,-1)).pow(1/self.p)

class GeMResNet(nn.Module):
    def __init__(self,backbone):
        super().__init__(); self.conv1=backbone.conv1; self.bn1=backbone.bn1; self.relu=backbone.relu; self.maxpool=backbone.maxpool
        self.layer1=backbone.layer1; self.layer2=backbone.layer2; self.layer3=backbone.layer3; self.layer4=backbone.layer4
        self.pool=GeM(); self.fc=nn.Sequential(nn.Linear(2048,512),nn.BatchNorm1d(512),nn.ReLU(inplace=True),nn.Dropout(.3),nn.Linear(512,1))
    def forward(self,x):
        for layer in (self.conv1,self.bn1,self.relu,self.maxpool,self.layer1,self.layer2,self.layer3,self.layer4): x=layer(x)
        return self.fc(self.pool(x))
def build_head(pretrained=True,pretrained_path=None): return GeMResNet(resnet(pretrained,pretrained_path))

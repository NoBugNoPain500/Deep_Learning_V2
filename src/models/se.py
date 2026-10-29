import torch
from torch import nn
from .base import resnet

class SEBlock(nn.Module):
    def __init__(self,channels,reduction=16):
        super().__init__(); self.pool=nn.AdaptiveAvgPool2d(1); self.fc=nn.Sequential(nn.Flatten(),nn.Linear(channels,channels//reduction),nn.ReLU(True),nn.Linear(channels//reduction,channels),nn.Sigmoid())
        nn.init.constant_(self.fc[3].bias,3.)
    def forward(self,x): return x*self.fc(self.pool(x)).unsqueeze(-1).unsqueeze(-1)

def build_se(pretrained=True,pretrained_path=None):
    model=resnet(pretrained,pretrained_path)
    for layer in (model.layer1,model.layer2,model.layer3,model.layer4):
        for block in layer:
            block.se=SEBlock(block.conv3.out_channels)
            old=block.forward
            def forward(x,old=old,b=block):
                identity=x
                out=b.conv1(x); out=b.bn1(out); out=b.relu(out); out=b.conv2(out); out=b.bn2(out); out=b.relu(out); out=b.conv3(out); out=b.bn3(out); out=b.se(out)
                if b.downsample is not None: identity=b.downsample(x)
                return b.relu(out+identity)
            block.forward=forward
    model.fc=nn.Sequential(nn.Dropout(.2),nn.Linear(2048,1)); return model

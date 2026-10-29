import torch
from torch import nn
from .base import resnet

class ChannelAttention(nn.Module):
    def __init__(self,c,r=16):
        super().__init__(); self.mlp=nn.Sequential(nn.Conv2d(c,c//r,1,bias=False),nn.ReLU(True),nn.Conv2d(c//r,c,1,bias=True)); nn.init.constant_(self.mlp[-1].bias,3.)
    def forward(self,x): return x*torch.sigmoid(self.mlp(torch.mean(x,(2,3),keepdim=True))+self.mlp(torch.amax(x,(2,3),keepdim=True)))
class CBAM(nn.Module):
    def __init__(self,c):
        super().__init__(); self.channel=ChannelAttention(c); self.spatial=nn.Conv2d(2,1,7,padding=3,bias=True); nn.init.constant_(self.spatial.bias,3.)
    def forward(self,x):
        x=self.channel(x); a=torch.cat((x.mean(1,keepdim=True),x.amax(1,keepdim=True)),1); return x*torch.sigmoid(self.spatial(a))
def build_cbam(pretrained=True,pretrained_path=None):
    model=resnet(pretrained,pretrained_path)
    for layer in (model.layer1,model.layer2,model.layer3,model.layer4):
        for block in layer:
            block.cbam=CBAM(block.conv3.out_channels)
            def forward(x,b=block):
                identity=x; out=b.relu(b.bn1(b.conv1(x))); out=b.relu(b.bn2(b.conv2(out))); out=b.bn3(b.conv3(out)); out=b.cbam(out)
                if b.downsample is not None: identity=b.downsample(x)
                return b.relu(out+identity)
            block.forward=forward
    model.fc=nn.Sequential(nn.Dropout(.2),nn.Linear(2048,1)); return model

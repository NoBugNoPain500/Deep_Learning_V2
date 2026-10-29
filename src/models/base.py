from __future__ import annotations
import torch
from torch import nn
from torchvision import models

def resnet(pretrained=True, pretrained_path=None, variant=None):
    weights=None
    if pretrained and not pretrained_path:
        try: weights=models.ResNet50_Weights.IMAGENET1K_V2
        except AttributeError: weights='DEFAULT'
    try: model=models.resnet50(weights=weights)
    except Exception as exc:
        if pretrained and not pretrained_path: raise RuntimeError('Cannot load ImageNet pretrained ResNet50. Enable Internet or set pretrained_path.') from exc
        model=models.resnet50(weights=None)
    if pretrained_path:
        state=torch.load(pretrained_path,map_location='cpu',weights_only=False)
        model.load_state_dict(state.get('state_dict',state),strict=False)
    if variant: variant(model)
    return model

def build_base(pretrained=True,pretrained_path=None):
    model=resnet(pretrained,pretrained_path)
    model.fc=nn.Sequential(nn.Dropout(.2),nn.Linear(2048,1))
    return model

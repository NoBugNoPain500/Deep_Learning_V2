from .base import build_base
from .head_variant import build_head
from .se import build_se
from .cbam import build_cbam

def build_model(name: str, pretrained: bool=True, pretrained_path=None):
    builders={'resnet50_base':build_base,'resnet50_head':build_head,'resnet50_se':build_se,'resnet50_cbam':build_cbam}
    if name not in builders: raise ValueError(f'Unknown model {name}; choose {list(builders)}')
    return builders[name](pretrained=pretrained,pretrained_path=pretrained_path)

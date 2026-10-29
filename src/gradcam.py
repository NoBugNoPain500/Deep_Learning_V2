"""Grad-CAM utility for ResNet layer4."""
import torch
class GradCAM:
    def __init__(self,model): self.model=model; self.activations=None; self.gradients=None; self.hook=model.layer4.register_forward_hook(self._forward)
    def _forward(self,module,args,out):
        self.activations=out; out.register_hook(lambda grad:setattr(self,'gradients',grad))
    def __call__(self,x,target=1):
        self.model.zero_grad(set_to_none=True); score=self.model(x).reshape(-1)[0]; (score if target else -score).backward()
        weights=self.gradients.mean((2,3),keepdim=True); cam=(weights*self.activations).sum(1,keepdim=True).relu(); cam=torch.nn.functional.interpolate(cam,x.shape[-2:],mode='bilinear',align_corners=False)
        return cam[0,0].detach()
    def close(self): self.hook.remove()

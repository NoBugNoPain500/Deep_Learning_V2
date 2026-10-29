import argparse,json
from pathlib import Path
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from .env import resolve_paths

def main():
    p=argparse.ArgumentParser(); p.add_argument('--models',nargs='+',default=['resnet50_base','resnet50_head','resnet50_se','resnet50_cbam']); p.add_argument('--seed',type=int,default=42); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--output_dir'); a=p.parse_args()
    import yaml
    cfg=yaml.safe_load(open(a.config)); cfg['output_dir']=a.output_dir or cfg.get('output_dir'); paths=resolve_paths(cfg); out=paths['output_dir']/'compare'; out.mkdir(parents=True,exist_ok=True); summary=[]; pred={}
    for name in a.models:
        run=paths['output_dir']/name/str(a.seed); m=json.loads((run/'evaluation/metrics.json').read_text()); hist=pd.read_csv(run/'history.csv'); fair=pd.read_csv(run/'evaluation/fairness_race.csv')
        summary.append({'model':name,'accuracy':m['accuracy'],'f1_macro':m['f1_macro'],'roc_auc':m['roc_auc'],'params':None,'race_accuracy_gap':fair.accuracy.max()-fair.accuracy.min(),'seconds_per_epoch':hist.epoch_seconds.mean()})
        pred[name]=pd.read_csv(run/'evaluation/predictions.csv')
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False); pd.DataFrame(summary).to_markdown(out/'summary.md',index=False)
    base=pred.get('resnet50_base')
    if base is not None:
        tests=[]
        for name,other in pred.items():
            if name=='resnet50_base': continue
            a0=(base.prediction!=base.y).to_numpy(); b0=(other.prediction!=other.y).to_numpy(); n01=int((a0&~b0).sum()); n10=int((~a0&b0).sum()); tests.append({'comparison':f'base vs {name}','discordant_base_only':n01,'discordant_variant_only':n10,'p_value':binomtest(min(n01,n10),n01+n10,.5).pvalue if n01+n10 else 1.})
        pd.DataFrame(tests).to_csv(out/'mcnemar.csv',index=False)
    fig,ax=plt.subplots()
    for name,frame in pred.items():
        fpr,tpr,_=roc_curve(frame.y,frame.probability); ax.plot(fpr,tpr,label=name)
    ax.plot([0,1],[0,1],'--',color='gray'); ax.set(xlabel='False positive rate',ylabel='True positive rate',title='FairFace test ROC'); ax.legend(); fig.savefig(out/'roc.png',dpi=180,bbox_inches='tight')
    print(pd.DataFrame(summary).to_string(index=False))
if __name__=='__main__': main()

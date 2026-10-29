import argparse,json,yaml
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,roc_auc_score,average_precision_score,confusion_matrix
from torch.utils.data import DataLoader
from .env import resolve_paths,find_image_root
from .dataset import FairFaceDataset
from .models import build_model

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--seed',type=int,default=42); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--data_dir'); p.add_argument('--output_dir'); a=p.parse_args()
    cfg=yaml.safe_load(open(a.config)); cfg['data_dir']=a.data_dir or cfg.get('data_dir'); cfg['output_dir']=a.output_dir or cfg.get('output_dir'); paths=resolve_paths(cfg)
    run=paths['output_dir']/a.model/str(a.seed); ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False); model=build_model(a.model,False).cpu(); model.load_state_dict(ck['model']); model.eval()
    df=pd.read_csv(paths['splits_dir']/'test.csv'); ds=FairFaceDataset(df,find_image_root(paths['test_csv'] or paths['data_dir'],df)); y=[]; probs=[]
    with torch.no_grad():
        for x,t,_ in DataLoader(ds,batch_size=32): y.extend(t.numpy()); probs.extend(torch.sigmoid(model(x)).flatten().numpy())
    y=np.array(y); probs=np.array(probs); pred=(probs>=.5).astype(int); pcls,rcls,fcls,_=precision_recall_fscore_support(y,pred,labels=[0,1],zero_division=0)
    metrics={'accuracy':accuracy_score(y,pred),'precision_macro':precision_recall_fscore_support(y,pred,average='macro',zero_division=0)[0],'recall_macro':precision_recall_fscore_support(y,pred,average='macro',zero_division=0)[1],'f1_macro':precision_recall_fscore_support(y,pred,average='macro',zero_division=0)[2],'roc_auc':roc_auc_score(y,probs),'pr_auc':average_precision_score(y,probs),'confusion_matrix':confusion_matrix(y,pred).tolist(),'precision_by_class':pcls.tolist(),'recall_by_class':rcls.tolist(),'f1_by_class':fcls.tolist()}
    out=run/'evaluation'; out.mkdir(exist_ok=True); (out/'metrics.json').write_text(json.dumps(metrics,indent=2)); pd.DataFrame({'y':y,'probability':probs,'prediction':pred,'race':df.race if 'race' in df else 'Unknown','age':df.age if 'age' in df else 'Unknown'}).to_csv(out/'predictions.csv',index=False)
    frame=pd.read_csv(out/'predictions.csv')
    for col in ('race','age'):
        rows=[]
        for group,g in frame.groupby(col,dropna=False):
            yy=g.y.to_numpy(); pp=g.prediction.to_numpy(); rows.append({'group':str(group),'n':len(g),'accuracy':accuracy_score(yy,pp),'f1':precision_recall_fscore_support(yy,pp,average='binary',zero_division=0)[2],'fpr':((pp==1)&(yy==0)).sum()/max(1,(yy==0).sum()),'fnr':((pp==0)&(yy==1)).sum()/max(1,(yy==1).sum())})
        pd.DataFrame(rows).to_csv(out/f'fairness_{col}.csv',index=False)
    print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()

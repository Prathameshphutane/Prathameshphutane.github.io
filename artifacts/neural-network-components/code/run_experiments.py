import numpy as np, pickle, json
from playground_port import run
ALL=('x1','x2','x1^2','x2^2','x1x2','sin(x1)','sin(x2)')
cfg={}
for d in ['gauss','circle','xor','spiral']: cfg[f'E1_{d}']=dict(data=d,epochs=300)
for nz in [0,0.25,0.5]: cfg[f'E2_noise{nz}']=dict(data='circle',noise=nz,epochs=300)
cfg['E3_circle_x_none']=dict(data='circle',shape=(),epochs=300)
cfg['E3_circle_sq_none']=dict(data='circle',feats=('x1^2','x2^2'),shape=(),epochs=300)
cfg['E3_spiral_all_42']=dict(data='spiral',feats=ALL,epochs=300)
for sh in [(),(2,),(4,2),(8,8),(8,8,8)]: cfg[f'E4_{sh}']=dict(data='spiral',shape=sh,epochs=1000)
for a in ['linear','sigmoid','relu','tanh']: cfg[f'E5_{a}']=dict(data='spiral',shape=(8,8,8),act=a,epochs=1000)
for lr in [0.003,0.03,0.3,3]: cfg[f'E6_lr{lr}']=dict(data='circle',lr=lr,epochs=300)
out={}
for k,c in cfg.items():
    rs=[run(seed=s,**c) for s in range(5)]
    te=[r['test_loss'] for r in rs]; tr=[r['train_loss'] for r in rs]; ac=[r['acc'] for r in rs]
    e=[r['epochs_to_01'] for r in rs if r['epochs_to_01']]
    out[k]=dict(test=float(np.median(te)),train=float(np.median(tr)),acc=float(np.median(ac)),reach=len(e),ep=float(np.median(e)) if e else None,params=rs[0]['params'],
               curve=np.median([[h[2] for h in r['hist']] for r in rs],0).tolist())
    print(k, round(out[k]['train'],3), round(out[k]['test'],3), round(out[k]['acc'],3), f"{len(e)}/5", out[k]['ep'], out[k]['params'])
json.dump(out,open('results.json','w'))

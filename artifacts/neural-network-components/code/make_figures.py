import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from playground_port import run, FEAT
import os; os.makedirs('img',exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
cm=LinearSegmentedColormap.from_list('pg',['#f59322','#ffffff','#0877bd'])  # playground orange/blue
def boundary(ax,r,title,sub):
    W,B,f,feats,Xte,Yte,D=r['model']
    g=np.linspace(-6,6,160); xx,yy=np.meshgrid(g,g)
    X=np.stack([FEAT[k](xx.ravel(),yy.ravel()) for k in feats],1); A=X
    for i in range(len(W)):
        z=A@W[i]+B[i]; A=np.tanh(z) if i==len(W)-1 else f(z)
    ax.contourf(xx,yy,A.reshape(xx.shape),levels=np.linspace(-1,1,21),cmap=cm,alpha=.85)
    k=len(D)//2; T=D[k:]
    ax.scatter(T[:,0],T[:,1],c=np.where(T[:,2]>0,'#0877bd','#f59322'),s=9,edgecolor='white',lw=.4)
    ax.set_xlim(-6,6); ax.set_ylim(-6,6); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title,fontsize=13,weight='bold',loc='left'); ax.text(0,-0.09,sub,transform=ax.transAxes,fontsize=10.5,color='#444',va='top')
    for s in ax.spines.values(): s.set_color('#ccc')
def panel(name,specs,seed_pick=None):
    fig,axs=plt.subplots(1,len(specs),figsize=(4.2*len(specs),4.6))
    for ax,(t,kw,s) in zip(axs,specs):
        r=run(**kw); boundary(ax,r,t,f"{s}\ntest loss {r['test_loss']:.3f} · accuracy {r['acc']*100:.0f}%")
    fig.tight_layout(); fig.savefig(f'img/{name}.png',dpi=150,bbox_inches='tight',transparent=False,facecolor='white'); plt.close(fig)
ALL=('x1','x2','x1^2','x2^2','x1x2','sin(x1)','sin(x2)')
panel('datasets',[('Gaussian',dict(data='gauss',epochs=300,seed=0),'[4,2] tanh, 300 epochs'),('Circle',dict(data='circle',epochs=300,seed=0),'[4,2] tanh, 300 epochs'),
  ('XOR',dict(data='xor',epochs=300,seed=0),'[4,2] tanh, 300 epochs'),('Spiral',dict(data='spiral',epochs=300,seed=0),'[4,2] tanh, 300 epochs')])
panel('noise',[('Noise 0',dict(data='circle',noise=0,epochs=300,seed=0),'Circle, [4,2] tanh'),('Noise 25',dict(data='circle',noise=.25,epochs=300,seed=0),'Circle, [4,2] tanh'),('Noise 50',dict(data='circle',noise=.5,epochs=300,seed=0),'Circle, [4,2] tanh')])
panel('capacity',[('[4,2] tanh',dict(data='spiral',shape=(4,2),epochs=1000,seed=0),'Spiral, 25 parameters'),('[8,8,8] tanh',dict(data='spiral',shape=(8,8,8),epochs=1000,seed=0),'Spiral, 177 parameters'),('[8,8,8] ReLU',dict(data='spiral',shape=(8,8,8),act='relu',epochs=1000,seed=0),'Spiral, 177 parameters')])
panel('features',[('X1, X2 only',dict(data='circle',shape=(),epochs=300,seed=0),'Circle, no hidden layer'),('X1², X2²',dict(data='circle',feats=('x1^2','x2^2'),shape=(),epochs=300,seed=0),'Circle, no hidden layer')])
R=json.load(open('results.json'))
fig,ax=plt.subplots(figsize=(8,4.6))
for k,c,l in [('E5_relu','#0877bd','ReLU'),('E5_tanh','#2a9d8f','Tanh'),('E5_sigmoid','#f59322','Sigmoid'),('E5_linear','#999','Linear')]:
    ax.plot(range(1,1001),R[k]['curve'],c=c,lw=2.2,label=l)
ax.set_xlabel('Epoch'); ax.set_ylabel('Test loss (median of 5 runs)'); ax.set_ylim(0,0.55); ax.legend(frameon=False)
for s in ['top','right']: ax.spines[s].set_visible(False)
ax.set_title('Spiral, [8,8,8] network: activation function vs. test loss',loc='left',fontsize=12)
fig.savefig('img/activation_curves.png',dpi=150,bbox_inches='tight',facecolor='white'); plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4.6))
for k,c,l in [('E6_lr0.003','#f59322','0.003 (too small)'),('E6_lr0.03','#2a9d8f','0.03 (default)'),('E6_lr0.3','#0877bd','0.3'),('E6_lr3','#c0392b','3 (too large)')]:
    ax.plot(range(1,301),R[k]['curve'],c=c,lw=2.2,label=l)
ax.set_xlabel('Epoch'); ax.set_ylabel('Test loss (median of 5 runs)'); ax.set_ylim(0,1.0); ax.legend(frameon=False,title='Learning rate')
for s in ['top','right']: ax.spines[s].set_visible(False)
ax.set_title('Circle, [4,2] tanh: learning rate vs. test loss',loc='left',fontsize=12)
fig.savefig('img/lr_curves.png',dpi=150,bbox_inches='tight',facecolor='white'); plt.close(fig)
print('ok')

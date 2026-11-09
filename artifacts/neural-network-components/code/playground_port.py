import numpy as np
# Faithful port of tensorflow/playground dataset.ts + nn.ts training loop
def circle(n,noise,R):
    pts=[]; rad=5
    for lo,hi in [(0,rad*.5),(rad*.7,rad)]:
        for _ in range(n//2):
            r=R.uniform(lo,hi); a=R.uniform(0,2*np.pi); x,y=r*np.sin(a),r*np.cos(a)
            nx,ny=R.uniform(-rad,rad)*noise,R.uniform(-rad,rad)*noise
            pts.append((x,y,1 if np.hypot(x+nx,y+ny)<rad*.5 else -1))
    return np.array(pts)
def xor(n,noise,R):
    pts=[]
    for _ in range(n):
        x=R.uniform(-5,5); x+=.3 if x>0 else -.3; y=R.uniform(-5,5); y+=.3 if y>0 else -.3
        nx,ny=R.uniform(-5,5)*noise,R.uniform(-5,5)*noise
        pts.append((x,y,1 if (x+nx)*(y+ny)>=0 else -1))
    return np.array(pts)
def gauss(n,noise,R):
    v=0.5+(4-0.5)*noise/0.5; s=np.sqrt(v)
    a=np.c_[R.normal(2,s,n//2),R.normal(2,s,n//2),np.ones(n//2)]
    b=np.c_[R.normal(-2,s,n//2),R.normal(-2,s,n//2),-np.ones(n//2)]
    return np.r_[a,b]
def spiral(n,noise,R):
    pts=[]; m=n//2
    for dt,lab in [(0,1),(np.pi,-1)]:
        for i in range(m):
            r=i/m*5; t=1.75*i/m*2*np.pi+dt
            pts.append((r*np.sin(t)+R.uniform(-1,1)*noise, r*np.cos(t)+R.uniform(-1,1)*noise, lab))
    return np.array(pts)
DATA={'circle':circle,'xor':xor,'gauss':gauss,'spiral':spiral}
FEAT={'x1':lambda x,y:x,'x2':lambda x,y:y,'x1^2':lambda x,y:x*x,'x2^2':lambda x,y:y*y,'x1x2':lambda x,y:x*y,'sin(x1)':lambda x,y:np.sin(x),'sin(x2)':lambda x,y:np.sin(y)}
ACT={'tanh':(np.tanh,lambda z,a:1-a*a),'relu':(lambda z:np.maximum(z,0),lambda z,a:(z>0).astype(float)),
     'sigmoid':(lambda z:1/(1+np.exp(-z)),lambda z,a:a*(1-a)),'linear':(lambda z:z,lambda z,a:np.ones_like(z))}
def run(data='circle',noise=0.0,feats=('x1','x2'),shape=(4,2),act='tanh',lr=0.03,batch=10,epochs=300,seed=0,n=500,track=None):
    R=np.random.default_rng(seed)
    D=DATA[data](n,noise,R); D=D[R.permutation(len(D))]; k=len(D)//2
    X=np.stack([FEAT[f](D[:,0],D[:,1]) for f in feats],1); Y=D[:,2]
    Xtr,Ytr,Xte,Yte=X[:k],Y[:k],X[k:],Y[k:]
    sizes=[len(feats),*shape,1]
    W=[R.uniform(-.5,.5,(sizes[i],sizes[i+1])) for i in range(len(sizes)-1)]
    B=[np.full(sizes[i+1],0.1) for i in range(len(sizes)-1)]
    f,df=ACT[act]
    def fwd(x):
        A=[x]; Z=[]
        for i in range(len(W)):
            z=A[-1]@W[i]+B[i]; Z.append(z); A.append(np.tanh(z) if i==len(W)-1 else f(z))
        return A,Z
    def loss(x,y):
        o=fwd(x)[0][-1][:,0]; return float(np.mean(0.5*(o-y)**2)), float(np.mean(np.sign(o+1e-12)==y))
    hist=[]; first=None
    for ep in range(1,epochs+1):
        for s in range(0,k-k%batch,batch):
            xb,yb=Xtr[s:s+batch],Ytr[s:s+batch]
            A,Z=fwd(xb); d=(A[-1][:,0]-yb)[:,None]*(1-A[-1]**2)
            for i in reversed(range(len(W))):
                gW=A[i].T@d/len(xb); gB=d.mean(0)
                if i>0: d=(d@W[i].T)*df(Z[i-1],A[i])
                W[i]-=lr*gW; B[i]-=lr*gB
            if not np.all(np.isfinite(W[0])): return dict(test_loss=float('nan'),train_loss=float('nan'),acc=float('nan'),epochs_to_01=None,hist=hist,diverged=True)
        tl,_=loss(Xtr,Ytr); te,acc=loss(Xte,Yte); hist.append((ep,tl,te))
        if first is None and te<0.1: first=ep
    return dict(train_loss=tl,test_loss=te,acc=acc,epochs_to_01=first,hist=hist,diverged=False,
                params=sum(w.size for w in W)+sum(b.size for b in B),model=(W,B,f,feats,Xte,Yte,D))

import numpy as np, pandas as pd, math, json, corrected_pipeline as cp
df=pd.read_csv("datasets_corrected/pakistan_dino.csv")
fc=[c for c in df.columns if c.startswith("feature_")]
X=df[fc].values.astype(np.float32); y=df["label"].values.astype(np.float32)
y=(y/1000.0)*(10000.0/(math.pi*17.5**2)); n=len(X)
GRID=[dict(k=k,cosine=c,simplex=True,l2=l2) for k in [3,5,8,10,12,15,20] for c in (True,False) for l2 in [0.0,0.1,1.0]]
cp.GRIDS["nnls"]=GRID

def nested_select(FC):
    preds=[]
    for i in range(n):
        m=np.ones(n,bool); m[i]=False; Xtr,ytr=X[m],y[m]; Xte=X[~m]
        K,cfg,cv,_=cp.select_K_and_cfg("nnls",Xtr,ytr,FC,5)
        idx=cp.rank_features_train(Xtr,ytr,K)
        preds.append(cp.predict_nnls(Xtr[:,idx],ytr,Xte[:,idx],**cfg)[0])
    mt=cp.metrics(y,preds); return mt["percent_rmse"],mt["r2"]

def nested_ensembleK(FC):
    # tune only cfg (neighbours/l2/metric) per K by CV, then AVERAGE predictions over all K
    preds=[]
    for i in range(n):
        m=np.ones(n,bool); m[i]=False; Xtr,ytr=X[m],y[m]; Xte=X[~m]
        pk=[]
        for K in FC:
            if K>Xtr.shape[1]: continue
            _,cfg,_,_=cp.select_K_and_cfg("nnls",Xtr,ytr,[K],5)  # best cfg at this K
            idx=cp.rank_features_train(Xtr,ytr,K)
            pk.append(cp.predict_nnls(Xtr[:,idx],ytr,Xte[:,idx],**cfg)[0])
        preds.append(float(np.mean(pk)))
    mt=cp.metrics(y,preds); return mt["percent_rmse"],mt["r2"]

for name,FC in {"apriori7 [20,40,60,100,150,200,260]":[20,40,60,100,150,200,260],
                "apriori6 [20,40,80,120,180,240]":[20,40,80,120,180,240]}.items():
    pr,r2=nested_select(FC); print(f"SELECT   {name:34s} -> %RMSE={pr:5.2f} R2={r2:.3f}",flush=True)
for name,FC in {"ensK full[10..360]":[c for c in cp.FEATURE_COUNTS if c<=X.shape[1]],
                "ensK coarse8":[20,40,60,80,100,140,200,260]}.items():
    pr,r2=nested_ensembleK(FC); print(f"ENSEMBLE {name:34s} -> %RMSE={pr:5.2f} R2={r2:.3f}",flush=True)
print("DIAG2_DONE")

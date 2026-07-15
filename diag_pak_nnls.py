import numpy as np, pandas as pd, math, json, corrected_pipeline as cp
df=pd.read_csv("datasets_corrected/pakistan_dino.csv")
fc=[c for c in df.columns if c.startswith("feature_")]
X=df[fc].values.astype(np.float32); y=df["label"].values.astype(np.float32)
y=(y/1000.0)*(10000.0/(math.pi*17.5**2)); n=len(X)
# simplex-only, wide-k, +l2 grid (the small-sample guard grid)
GRID=[dict(k=k,cosine=c,simplex=True,l2=l2) for k in [3,5,8,10,12,15,20] for c in (True,False) for l2 in [0.0,0.1,1.0]]
cp.GRIDS["nnls"]=GRID
def nested(FC):
    preds=[]; Ks=[]
    for i in range(n):
        m=np.ones(n,bool); m[i]=False
        Xtr,ytr=X[m],y[m]; Xte=X[~m]
        K,cfg,cv,_=cp.select_K_and_cfg("nnls",Xtr,ytr,FC,5)
        idx=cp.rank_features_train(Xtr,ytr,K)
        preds.append(cp.predict_nnls(Xtr[:,idx],ytr,Xte[:,idx],**cfg)[0]); Ks.append(K)
    mt=cp.metrics(y,preds); return mt["percent_rmse"],mt["r2"],Ks
FCs={
 "full 18 [10..360]":[c for c in cp.FEATURE_COUNTS if c<=X.shape[1]],
 "coarse8 [20..260]":[20,40,60,80,100,140,200,260],
 "coarse5 [20,50,100,150,200]":[20,50,100,150,200],
 "coarse4 [30,60,100,150]":[30,60,100,150],
 "cap<=160 [10..160]":[10,20,30,40,50,60,70,80,90,100,120,140,160],
 "cap<=120 coarse":[20,40,60,90,120],
}
out={}
for name,FC in FCs.items():
    pr,r2,Ks=nested(FC)
    out[name]={"prmse":pr,"r2":r2,"Kspread":[min(Ks),max(Ks)]}
    print(f"{name:32s} -> %RMSE={pr:5.2f} R2={r2:.3f}  K in [{min(Ks)},{max(Ks)}]",flush=True)
json.dump(out,open("results_corrected/logs/diag_pak_nnls.json","w"),indent=2)
print("DIAG_DONE")

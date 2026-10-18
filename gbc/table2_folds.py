# Table 2 quantities for f1 with the label coded as the zero-one loss, per-fold values for f2, and constant-risk
# baselines across search measures (App. J).
import json, glob, os, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
F=np.load(f"{ROOT}/gbc/features.npz"); M=["bfs","dfs","mixed"]
def labels(pattern,m):
    y={}
    for p in glob.glob(f"{ROOT}/gbc/labels/{pattern}_{m}*.jsonl"):
        for line in open(p):
            r=json.loads(line); y[r["idx"]]=r["correct"]
    idx=np.array(sorted(y)); return idx,np.array([y[i] for i in idx])
def ll(y,p):
    p=np.clip(p,1e-15,1-1e-15); return float(np.sum(y*np.log(p)+(1-y)*np.log(1-p)))
def cv(X,y):
    A=[];R=[]
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(X,y):
        p=HistGradientBoostingClassifier(random_state=0).fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
        A.append(roc_auc_score(y[te],p)); R.append(1-ll(y[te],p)/ll(y[te],np.full(len(te),y[tr].mean())))
    return dict(auc=float(np.mean(A)),mcf=float(np.mean(R)),auc_folds=[round(a,4) for a in A],mcf_folds=[round(r,4) for r in R])
out={}
for tag,pat in [("f1","f1_small_deep_mixed_rope"),("f2","f2_large_deep_bfs_ape")]:
    X={};Y={};T={}
    for m in M:
        idx,y=labels(pat,m); X[m]={"F12":F[f"{m}_F12"][idx],"LEN":F[f"{m}_LEN"][idx],"PROB":F[f"{m}_PROB"][idx]}; Y[m]=y; T[m]=F[f"{m}_in_any_train"][idx]
    y=np.concatenate([Y[m] for m in M]); intr=np.concatenate([T[m] for m in M])
    F12=np.concatenate([X[m]["F12"] for m in M]); LEN=np.concatenate([X[m]["LEN"] for m in M]); PROB=np.concatenate([X[m]["PROB"] for m in M])
    r={}
    for coding,yy in ([("loss",1-y)] if tag=="f1" else [("impl",y)]):
        if True:
            r[f"F11_{coding}"]=cv(np.delete(F12,9,axis=1),yy)
            r[f"notin_{coding}"]=cv(F12[~intr],yy[~intr])
            for m in M: r[f"within_{m}_{coding}"]=cv(X[m]["F12"],1-Y[m] if coding=="loss" else Y[m])
        if tag=="f2":
            r[f"F12_{coding}"]=cv(F12,yy); r[f"LEN_{coding}"]=cv(LEN,yy); r[f"PROB_{coding}"]=cv(PROB,yy)
    # constant transfer baseline: fit constant on S, apply to T
    risk={m:1-Y[m].mean() for m in M}
    r["const_transfer_max"]=max(abs(risk[s]-risk[t]) for s in M for t in M if s!=t)
    pooled=1-y.mean(); r["const_pooled_max"]=max(abs(risk[m]-pooled) for m in M)
    out[tag]=r; print(tag,json.dumps(r,indent=1),flush=True)
json.dump(out,open(f"{ROOT}/gbc/results_table2_folds.json","w"),indent=1)

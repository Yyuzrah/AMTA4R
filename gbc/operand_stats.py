# App. J: operand-count baseline, accuracy and training overlap by operand count, labels of repeated prompts,
# largest cell deviation under perfect calibration, and the cell-level R2 of the operand count.
import json, glob, os, numpy as np
from collections import defaultdict
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
    oof=np.zeros(len(y)); A=[];R=[]
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(X,y):
        p=HistGradientBoostingClassifier(random_state=0).fit(X[tr],y[tr]).predict_proba(X[te])[:,1]; oof[te]=p
        A.append(roc_auc_score(y[te],p)); R.append(1-ll(y[te],p)/ll(y[te],np.full(len(te),y[tr].mean())))
    return np.mean(A),np.std(A),np.mean(R),np.std(R),oof
fresh={m:json.load(open(f"{ROOT}/gbc/data/fresh_{m}.json")) for m in M}
out={}
for tag,pat in [("f1","f1_small_deep_mixed_rope"),("f2","f2_large_deep_bfs_ape")]:
    ys=[];nops=[];intr=[];meas=[];tgt=[];prompts=[];P=[]
    for m in M:
        idx,y=labels(pat,m); ys.append(y); pr=F[f"{m}_PROB"][idx]; P.append(pr); nops.append(pr[:,1]); tgt.append(pr[:,0]); intr.append(F[f"{m}_in_any_train"][idx]); meas+= [m]*len(y)
        prompts += [(fresh[m][i]["target"],tuple(fresh[m][i]["nums"])) for i in idx]
    y=np.concatenate(ys); nops=np.concatenate(nops); intr=np.concatenate(intr); tgt=np.concatenate(tgt); meas=np.array(meas)
    r={}
    for coding,yy in [("correct",y),("loss",1-y)]:
        a,sa,rr,sr,oof=cv(nops.reshape(-1,1),yy)
        r[f"nops_{coding}"]=dict(auc=a,auc_sd=sa,mcf=rr,mcf_sd=sr,pooled_auc=roc_auc_score(yy,oof), pooled_mcf=1-ll(yy,oof)/ll(yy,np.full(len(yy),yy.mean())))
    for k in (3,4):
        s=nops==k; r[f"acc_{k}ops"]=(float(y[s].mean()), int(s.sum())); r[f"intrain_frac_{k}ops"]=float(intr[s].mean())
    r["overlap_total"]=float(intr.mean()); r["overlap_share_3ops"]=float((nops[intr]==3).mean()); r["notintrain_share_4ops"]=float((nops[~intr]==4).mean())
    # determinism: duplicates of exact prompt
    g=defaultdict(list)
    for p,l in zip(prompts,y): g[p].append(l)
    dup=[v for v in g.values() if len(v)>1]
    r["dup_groups"]=len(dup); r["dup_inconsistent"]=sum(1 for v in dup if len(set(v))>1); r["dup_items"]=sum(len(v) for v in dup)
    # sorted-prompt duplicates
    g2=defaultdict(list)
    for p,l in zip(prompts,y): g2[(p[0],tuple(sorted(p[1])))].append(l)
    dup2=[v for v in g2.values() if len(v)>1]; r["sorted_dup_groups"]=len(dup2); r["sorted_dup_inconsistent"]=sum(1 for v in dup2 if len(set(v))>1)
    # largest cell deviation when labels are drawn from the out-of-fold chi predictions (cells as in subpopulations.py)
    X12=np.concatenate([F[f"{m}_F12"][labels(pat,m)[0]] for m in M])
    a,sa,rr,sr,oof=cv(X12,y)
    edges=list(range(10,101,10)) if tag=="f1" else [10,40,70,100]
    band=np.digitize(tgt,edges[1:-1]); cell=np.array([f"{a}|{b}|{c}" for a,b,c in zip(meas,nops.astype(int),band)])
    ks,ci=np.unique(cell,return_inverse=True); rng=np.random.default_rng(0); mx=[]
    cnt=np.bincount(ci); pm=np.bincount(ci,weights=oof)/cnt
    for _ in range(4000):
        ysim=(rng.random(len(oof))<oof).astype(float)
        mx.append(np.max(np.abs(np.bincount(ci,weights=ysim)/cnt-pm)))
    r["noise_floor_p95"]=float(np.percentile(mx,95)); r["noise_floor_median"]=float(np.median(mx)); r["ncells"]=len(ks)
    Rk=np.bincount(ci,weights=y)/cnt
    # operand count R2_meas
    _,_,_,_,oofn=cv(nops.reshape(-1,1),y); Rh=np.bincount(ci,weights=oofn)/cnt
    r["r2meas_nops"]=float(1-np.sum((Rk-Rh)**2)/np.sum((Rk-Rk.mean())**2))
    out[tag]=r; print(tag, json.dumps(r,indent=1,default=float), flush=True)
json.dump(out,open(f"{ROOT}/gbc/results_operand_stats.json","w"),indent=1,default=float)

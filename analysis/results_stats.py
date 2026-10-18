# Derived statistics quoted in the paper (Sec. 5, App. G.3, H, I, K), from results/final_results_54.json.
import json, itertools, math
R=json.load(open("results/final_results_54.json"))
print("n configs", len(R))
arch={("small","deep"):(12,192),("small","balanced"):(6,256),("small","wide"):(3,384),
("medium","deep"):(24,512),("medium","balanced"):(12,768),("medium","wide"):(6,1024),
("large","deep"):(48,768),("large","balanced"):(24,1024),("large","wide"):(12,1536)}
S=["small","medium","large"]; SH=["deep","balanced","wide"]; TR=["bfs","dfs","mixed"]; PE=["ape","rope"]; EV=["bfs","dfs","mixed"]
def a(s,sh,tr,pe,ev): return R[f"{s}_{sh}_{tr}_{pe}"][ev]
p=lambda *x: print(*x)
p("Sec5.2: LWMixed APE mixed", a("large","wide","mixed","ape","mixed"), "RoPE", a("large","wide","mixed","rope","mixed"))
p("LDMixed APE mixed", a("large","deep","mixed","ape","mixed"), "MBMixed APE mixed", a("medium","balanced","mixed","ape","mixed"))
p("Fig1B: LW DFS APE dfs", a("large","wide","dfs","ape","dfs"), "LD DFS APE dfs", a("large","deep","dfs","ape","dfs"))
p("H1/Fig2A: APE DFS-trained on dfs large wide/bal/deep", [a("large",sh,"dfs","ape","dfs") for sh in ["wide","balanced","deep"]],
  "RoPE", [a("large",sh,"dfs","rope","dfs") for sh in ["wide","balanced","deep"]])
p("Fig2B Medium DFS-trained dfs: APE W,B,D", [a("medium",sh,"dfs","ape","dfs") for sh in ["wide","balanced","deep"]],
  "RoPE", [a("medium",sh,"dfs","rope","dfs") for sh in ["wide","balanced","deep"]])
# Fig. 2B comparison for all training measures and evaluation sets
for tr in TR:
  for ev in EV:
    ap=[a("medium",sh,tr,"ape",ev) for sh in ["wide","balanced","deep"]]; ro=[a("medium",sh,tr,"rope",ev) for sh in ["wide","balanced","deep"]]
    p(f"  medium tr={tr} ev={ev} APE {ap} RoPE {ro} diffs {[round(100*(r-x),2) for r,x in zip(ro,ap)]}")
# G.3 width comparisons: best of six per architecture on DFS
best={}
for (s,sh),(L,d) in arch.items():
    vals=[a(s,sh,tr,pe,"dfs") for tr in TR for pe in PE]
    best[(L,d)]=max(vals)
p("G3 best of six on DFS:", {k:v for k,v in sorted(best.items())})
for (L,d1,d2) in [(12,192,768),(12,768,1536),(24,512,1024),(6,256,1024)]:
    p(f"  L={L} d {d1}->{d2}: {best[(L,d1)]:.4f}->{best[(L,d2)]:.4f} diff {100*(best[(L,d2)]-best[(L,d1)]):.2f} pts")
# model-by-model doublings from d>=512
inv={v:k for k,v in arch.items()}
for pe in PE:
  ds=[]
  for (L,d1,d2) in [(12,768,1536),(24,512,1024)]:
    for tr in TR:
      x=a(*inv[(L,d1)],tr,pe,"dfs"); y=a(*inv[(L,d2)],tr,pe,"dfs"); ds.append(round(100*(y-x),2))
  p(f"  doublings {pe}: {ds} min {min(ds)} max {max(ds)}")
# App I: best accuracies at L=12,24,48 per eval set
Lgroups={12:[("small","deep"),("medium","balanced"),("large","wide")],24:[("medium","deep"),("large","balanced")],48:[("large","deep")],6:[("small","balanced"),("medium","wide")],3:[("small","wide")]}
for ev in EV:
    bl={L:max(a(s,sh,tr,pe,ev) for (s,sh) in g for tr in TR for pe in PE) for L,g in Lgroups.items()}
    v=[bl[12],bl[24],bl[48]]
    p(f"AppI ev={ev}: best L=12 {bl[12]} L=24 {bl[24]} L=48 {bl[48]} spread {100*(max(v)-min(v)):.2f} pts | L=6 {bl[6]} L=3 {bl[3]}")
# same, separately for APE and RoPE
for ev in EV:
  for pe in PE:
    bl={L:max(a(s,sh,tr,pe,ev) for (s,sh) in g for tr in TR) for L,g in Lgroups.items()}
    v=[bl[12],bl[24],bl[48]]
    p(f"  per PE {pe} ev={ev}: {bl[12]},{bl[24]},{bl[48]} spread {100*(max(v)-min(v)):.2f}")
# K.5 averages per training measure over 18 models
for tr in TR:
    m={ev: sum(a(s,sh,tr,pe,ev) for s in S for sh in SH for pe in PE)/18 for ev in EV}
    p(f"K5 {tr}-trained: bfs {m['bfs']:.4f} mixed {m['mixed']:.4f} dfs {m['dfs']:.4f} drop bfs->dfs {100*(m['bfs']-m['dfs']):.2f}")
p("K5 LD BFS APE eval: bfs", a("large","deep","bfs","ape","bfs"), "dfs", a("large","deep","bfs","ape","dfs"))
# Sec 5 RoPE vs APE
wins=0; tot=0; pw=0; gaps={s:[] for s in S}; large_w=0; large_t=0
for s in S:
  for sh in SH:
    for tr in TR:
      for ev in EV:
        x=a(s,sh,tr,"ape",ev); y=a(s,sh,tr,"rope",ev); tot+=1; wins+= y>x; gaps[s].append(y-x)
        if s=="large": large_t+=1; large_w+= y>x
      ma=sum(a(s,sh,tr,"ape",ev) for ev in EV); mr=sum(a(s,sh,tr,"rope",ev) for ev in EV); pw+= mr>ma
p(f"RoPE>APE {wins}/{tot}; model pairs {pw}/27; large {large_w}/{large_t}; mean gaps", {s: round(100*sum(g)/len(g),2) for s,g in gaps.items()})
p("sign test p(>=25/27)", sum(math.comb(27,k) for k in range(25,28))/2**27)
# Deep vs Wide under APE, medium/large (18 comparisons)
c=0;n=0
for s in ["medium","large"]:
  for tr in TR:
    for ev in EV:
      n+=1; c+= a(s,"deep",tr,"ape",ev)>a(s,"wide",tr,"ape",ev)
p(f"APE deep>wide medium/large {c}/{n}")
# RoPE large: |deep-wide| <1pt in nine comparisons ; APE range
dr=[abs(a("large","deep",tr,"rope",ev)-a("large","wide",tr,"rope",ev)) for tr in TR for ev in EV]
da=[a("large","deep",tr,"ape",ev)-a("large","wide",tr,"ape",ev) for tr in TR for ev in EV]
p("large RoPE |deep-wide| max pts", round(100*max(dr),2), "APE deep-wide range", round(100*min(da),2), round(100*max(da),2))
# H.2: Large APE in-distribution accuracy, Wide -> Balanced -> Deep, per training measure
for tr in TR:
    w=a("large","wide",tr,"ape",tr); b=a("large","balanced",tr,"ape",tr); dd=a("large","deep",tr,"ape",tr)
    p(f"H2 large APE ID tr={tr}: wide {w} bal {b} deep {dd} rise {100*(dd-w):.2f}")
# monotone rise all train/eval at Large APE
mono=all(a("large","wide",tr,"ape",ev)<a("large","balanced",tr,"ape",ev)<a("large","deep",tr,"ape",ev) for tr in TR for ev in EV)
mono2=all(a("large","wide",tr,"ape",ev)<=a("large","balanced",tr,"ape",ev)<=a("large","deep",tr,"ape",ev) for tr in TR for ev in EV)
p("Large APE strictly monotone in depth all tr/ev:", mono, "non-strict:", mono2)

"""Shared helpers: result directory of a unit (results/rescue/ if gatecheck.py selected the unit for a strict re-run
and that re-run finished, else results/raw/) and record loading."""
import json, os
from registry import units, scen_tasks

REP_MAX = 0.002       # largest replicate-row answer-flip rate per task (1 in 500)
V4_MIN = 0.99         # smallest agreement of cached decoding with full recomputation (argmax of the 3rd token)


def unit_dirs(base, u):
    uid, tag, scen = units()[u]
    return f'{base}/results/raw/{tag}/{scen}', f'{base}/results/rescue/{tag}/{scen}'


def effective_dir(base, u, rescue_set):
    raw, res = unit_dirs(base, u)
    if u in rescue_set and os.path.exists(f'{res}/DONE.json'):
        return res
    if os.path.exists(f'{raw}/DONE.json'):
        return raw
    return None


def load_unit(d, scen):
    recs = {}
    for t in scen_tasks(scen):
        p = f'{d}/{t}.jsonl'
        if os.path.exists(p):
            recs[t] = [json.loads(l) for l in open(p)]
    return recs


def rescue_set(base):
    p = f'{base}/results/gatecheck.json'
    return set(json.load(open(p))['rescue']) if os.path.exists(p) else set()

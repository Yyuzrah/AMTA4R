"""After all units have run (CPU): list the fp32 units that get one strict re-run (pc_run.py --strict, every row in
its own forward pass): units whose replicate row changes the answer in more than 0.2% of the items of any task, whose
cached decoding agrees with full recomputation in less than 99% of the checks, with non-finite logits, or without a
finished result. Only these validity quantities are used, never the rows with shifted position indices. Writes
results/gatecheck.json (key "rescue": units to re-run) and the empty file results/markers/rescue_<u>.done of every
unit that is not re-run.
Usage: gatecheck.py <base>"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from registry import units, FP32_SCEN
from common_io import unit_dirs, load_unit, REP_MAX, V4_MIN


def main():
    base = sys.argv[1]
    out = f'{base}/results/gatecheck.json'
    if os.path.exists(out):
        print('done marker exists', out); return 0
    os.makedirs(f'{base}/results/markers', exist_ok=True)
    rescue, reasons, ok = [], {}, {}
    for u, (uid, tag, scen) in enumerate(units()):
        raw, _ = unit_dirs(base, u)
        if scen not in FP32_SCEN:                       # only fp32 units are re-run; bf16 units are reported as run
            ok[uid] = 'bf16 secondary unit: no re-run by design'
            continue
        if not os.path.exists(f'{raw}/DONE.json'):
            if os.path.exists(f'{base}/results/markers/notrunnable_{u}'):
                ok[uid] = 'unit not runnable (model not in the item plan)'
            elif os.path.exists(f'{base}/meta/fetch/{tag}.json'):
                rescue.append(u); reasons[uid] = ['no finished run result']
            else:
                ok[uid] = 'model not fetched; unit cannot run'
            continue
        why = []
        ck = json.load(open(f'{raw}/checks.json'))['summary']
        if (ck.get('V4_cached_vs_full_agree') if ck.get('V4_cached_vs_full_agree') is not None else 1.0) < V4_MIN:
            why.append(f"V4 {ck['V4_cached_vs_full_agree']:.4f} < {V4_MIN}")
        if not ck.get('finite', True):
            why.append('non-finite logits in checks')
        recs = load_unit(raw, scen)
        for t, rr in recs.items():
            rep = sum(r['ans'][r['labels'].index('rep')] != r['ans'][r['labels'].index('ref')] for r in rr) / len(rr)
            if rep > REP_MAX:
                why.append(f'{t}: replicate flip rate {rep:.4f} > {REP_MAX}')
            if not all(r.get('finite', True) for r in rr):
                why.append(f'{t}: non-finite first-step logits')
        if why:
            rescue.append(u); reasons[uid] = why
        else:
            ok[uid] = 'gates G3/finite pass; no re-run'
    for u, (uid, tag, scen) in enumerate(units()):
        if u not in rescue:
            m = f'{base}/results/markers/rescue_{u}.done'
            open(m, 'w').write(f'{uid}: {ok.get(uid, "no re-run")}\n')
    res = {'time': time.strftime('%Y-%m-%dT%H:%M:%S'), 'rescue': rescue,
           'rescue_uids': [units()[u][0] for u in rescue], 'reasons': reasons, 'not_rescued': ok}
    json.dump(res, open(out + '.tmp', 'w'), indent=1); os.replace(out + '.tmp', out)
    print('rescue units:', len(rescue), res['rescue_uids'])
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""Selects the cells that are re-trained with STEPS_EXT = 60,000 steps, from the 30,000-step runs in BASE/runs/.
Gate G_a of a run: test A_single(d) >= 0.9 for every d in 1..d_train (in distribution; splitting is not used).
If any run of a (d_train, model, PE) cell fails G_a, all seeds of the cell are re-trained once with STEPS_EXT steps
into BASE/runs_ext/, and analyze.py uses those runs for the cell. Missing runs are listed and not re-trained.
Writes BASE/runs/gatecheck.json (rescue_names, and rescue_idx = the grid rows to re-train).
Usage: python gatecheck.py BASE GRID_TSV"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

base, grid_path = sys.argv[1], sys.argv[2]
grid = C.load_grid(grid_path)
cells = {}
for r in grid:
    cells.setdefault(C.cell_key(r), []).append(r)
per_run, missing = {}, []
for r in grid:
    res = C.read_json(os.path.join(base, 'runs', r['name'], 'results.json'))
    if res is None:
        missing.append(r['name']); continue
    per_run[r['name']] = dict(gate_a=bool(C.gate_a(res)),
                              A_in=[round(res['test_single']['per_depth'][str(d)]['seq'], 4)
                                    for d in range(1, r['dtrain'] + 1)])
rescue_cells = [k for k, rows in cells.items()
                if any(x['name'] in per_run and not per_run[x['name']]['gate_a'] for x in rows)]
rescue_names = sorted(x['name'] for k in rescue_cells for x in cells[k])
rescue_idx = sorted(x['idx'] for k in rescue_cells for x in cells[k])
out = dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'), n_grid=len(grid), n_found=len(per_run), missing=missing,
           per_run=per_run, rescue_cells=[list(k) for k in rescue_cells], rescue_names=rescue_names,
           rescue_idx=rescue_idx, steps_ext=C.STEPS_EXT)
tmp = os.path.join(base, 'runs', 'gatecheck.json.tmp')
json.dump(out, open(tmp, 'w'), indent=1)
os.replace(tmp, os.path.join(base, 'runs', 'gatecheck.json'))
print(f"gatecheck: {len(per_run)}/{len(grid)} runs found; G_a failures: "
      f"{[n for n, v in per_run.items() if not v['gate_a']]}; cells re-run with 2x steps: {rescue_cells}; missing: {missing}")

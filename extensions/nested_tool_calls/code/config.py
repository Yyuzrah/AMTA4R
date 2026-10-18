"""Design constants of the nested tool-call experiment.
Quick test mode (env NTC_QUICK_TEST=1) only shrinks the sample sizes and writes under ../smoke/."""
import os

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE = os.environ.get('NTC_QUICK_TEST', '0') == '1'
BASE = os.environ.get('NTC_BASE') or (os.path.join(EXP, 'smoke') if SMOKE else EXP)   # holds results/, analysis/
# (NTC_BASE: another folder; default: the folder above code/, or its smoke/ in quick test mode)
RES = os.path.join(BASE, 'results')
ANA = os.path.join(BASE, 'analysis')
MARK = os.path.join(BASE, 'markers')

DEPTHS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]     # nesting depth grid
M_FUN = 24                      # line budget = max depth; an item has d chain tools + (M_FUN - d) note lines ...
DIS_MODE = 'k2'                 # ... + 2 distractor tools that never call a chain tool (26 registry lines in all)
N_TOOLS_EXTRA = 2
ORDER = 'callee'
TASKS = ['T']                   # T: nested tool-call returns (tasks2.py also defines a to-do surface S, not run)
PRIMARY = 'T'
SPLIT_CAL, SPLIT_TEST = ('smokecal', 'smoketest') if SMOKE else ('cal', 'test')
N_CAL = 100                     # items per depth, calibration split
N_TEST = 200                    # items per depth, test split
if SMOKE:                       # quick test mode only
    N_CAL = int(os.environ.get('NTC_N_CAL', 4))
    N_TEST = int(os.environ.get('NTC_N_TEST', 4))
RERUN_FRAC = 0.10               # re-run check: fraction of I_test items re-run in a fresh process
MAXTOK = 8                      # answer-only generation budget (integer answer after the 'Answer:' prefill)
MAX_MODEL_LEN = 4096

# Thresholds used by analyze2.py. B_m: the test depths deeper than d_hat* of model m.
TAU = 0.80                      # d_hat* = largest depth whose fit of the cal accuracy (not increasing with
                                # depth) is >= TAU
TAU_SENS = 0.90                 # second value of TAU
TAUS = (TAU, TAU_SENS)
INCL_DEPTHS = (1, 2)            # flag 'included': cal accuracy of I at d = 1 and d = 2 both >= TAU
TOL_A = 0.02                    # check (b): accuracy of H at most 2 points below A on B_m
MARGIN_I = 0.10                 # check (c): H at least 10 points above I on B_m, paired bootstrap 95% lower bound > 0
NOHARM = 0.05                   # check noharm: mean H over all depths >= mean A over all depths - 0.05
GATE_MIN_MODELS = 4             # flag 'gate': >= 4 included models ...
GATE_MIN_FAMILIES = 2           # ... from >= 2 families ...
GATE_BIG = 13                   # ... including >= 1 model with >= 13B parameters
BOOT = 200 if SMOKE else 2000

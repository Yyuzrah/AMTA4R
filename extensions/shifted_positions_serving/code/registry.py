"""Models, scenarios, tasks, shifts of the position indices and paths of the serving experiment.
MODELS[tag] = (hub id, revision or None, position encoding, HF model family, trained context, license, size class).
Revisions: OPT and GPT-J publish only pickle (.bin) weights on `main`, which transformers 4.57.1 with torch 2.5.1
refuses to load (CVE-2025-32434), so the safetensors conversions that the Hugging Face conversion bot (SFconvertbot)
opened as pull requests are used (the earliest such pull request of each model). fetch.py records the downloaded
commit in meta/fetch/<tag>.json."""
import os

E = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # folder of this experiment
FEWSHOT_DIR = os.environ.get('FEWSHOT_DIR', f'{E}/fewshot')      # output folder of make_fewshot_items.py

MIT, APACHE = 'MIT', 'Apache-2.0'
OPT_LIC = 'OPT-175B License Agreement (hub tag "other"; non-commercial research use permitted)'
BLOOM_LIC = 'BigScience BLOOM RAIL License v1.0 (open RAIL; research use permitted with use restrictions)'
QWEN_RES = 'Qwen Research License (hub tag "other"; non-commercial research use permitted)'
OLMO = 'Apache-2.0'

MODELS = {
    # learned absolute position embeddings (APE)
    'gpt2':          ('gpt2', None, 'APE', 'gpt2', 1024, MIT, 'S'),
    'gpt2-medium':   ('gpt2-medium', None, 'APE', 'gpt2', 1024, MIT, 'S'),
    'gpt2-large':    ('gpt2-large', None, 'APE', 'gpt2', 1024, MIT, 'S'),
    'gpt2-xl':       ('gpt2-xl', None, 'APE', 'gpt2', 1024, MIT, 'S'),
    'gpt-neo-125m':  ('EleutherAI/gpt-neo-125m', None, 'APE', 'gpt_neo', 2048, MIT, 'S'),
    'gpt-neo-1.3B':  ('EleutherAI/gpt-neo-1.3B', None, 'APE', 'gpt_neo', 2048, MIT, 'S'),
    'gpt-neo-2.7B':  ('EleutherAI/gpt-neo-2.7B', None, 'APE', 'gpt_neo', 2048, MIT, 'S'),
    'opt-125m':      ('facebook/opt-125m', 'refs/pr/35', 'APE', 'opt', 2048, OPT_LIC, 'S'),
    'opt-350m':      ('facebook/opt-350m', 'refs/pr/33', 'APE', 'opt', 2048, OPT_LIC, 'S'),
    'opt-1.3b':      ('facebook/opt-1.3b', 'refs/pr/25', 'APE', 'opt', 2048, OPT_LIC, 'S'),
    'opt-2.7b':      ('facebook/opt-2.7b', 'refs/pr/24', 'APE', 'opt', 2048, OPT_LIC, 'S'),
    'opt-6.7b':      ('facebook/opt-6.7b', 'refs/pr/31', 'APE', 'opt', 2048, OPT_LIC, 'L'),
    # ALiBi
    'bloom-560m':    ('bigscience/bloom-560m', None, 'ALiBi', 'bloom', 2048, BLOOM_LIC, 'S'),
    'bloom-1b1':     ('bigscience/bloom-1b1', None, 'ALiBi', 'bloom', 2048, BLOOM_LIC, 'S'),
    'bloom-1b7':     ('bigscience/bloom-1b7', None, 'ALiBi', 'bloom', 2048, BLOOM_LIC, 'S'),
    'bloom-3b':      ('bigscience/bloom-3b', None, 'ALiBi', 'bloom', 2048, BLOOM_LIC, 'S'),
    'bloom-7b1':     ('bigscience/bloom-7b1', None, 'ALiBi', 'bloom', 2048, BLOOM_LIC, 'L'),
    # RoPE
    'pythia-160m':   ('EleutherAI/pythia-160m', None, 'RoPE', 'gpt_neox', 2048, APACHE, 'S'),
    'pythia-410m':   ('EleutherAI/pythia-410m', None, 'RoPE', 'gpt_neox', 2048, APACHE, 'S'),
    'pythia-1b':     ('EleutherAI/pythia-1b', None, 'RoPE', 'gpt_neox', 2048, APACHE, 'S'),
    'pythia-1.4b':   ('EleutherAI/pythia-1.4b', None, 'RoPE', 'gpt_neox', 2048, APACHE, 'S'),
    'pythia-2.8b':   ('EleutherAI/pythia-2.8b', None, 'RoPE', 'gpt_neox', 2048, APACHE, 'S'),
    'pythia-6.9b':   ('EleutherAI/pythia-6.9b', None, 'RoPE', 'gpt_neox', 2048, APACHE, 'L'),
    'gpt-j-6b':      ('EleutherAI/gpt-j-6b', 'refs/pr/40', 'RoPE', 'gptj', 2048, APACHE, 'L'),
    'qwen2.5-0.5b':  ('Qwen/Qwen2.5-0.5B', None, 'RoPE', 'qwen2', 32768, APACHE, 'S'),
    'qwen2.5-1.5b':  ('Qwen/Qwen2.5-1.5B', None, 'RoPE', 'qwen2', 32768, APACHE, 'S'),
    'qwen2.5-3b':    ('Qwen/Qwen2.5-3B', None, 'RoPE', 'qwen2', 32768, QWEN_RES, 'S'),
    'qwen2.5-7b':    ('Qwen/Qwen2.5-7B', None, 'RoPE', 'qwen2', 32768, APACHE, 'L'),
    'olmo-2-1b':     ('allenai/OLMo-2-0425-1B', None, 'RoPE', 'olmo2', 4096, OLMO, 'S'),
    'olmo-2-7b':     ('allenai/OLMo-2-1124-7B', None, 'RoPE', 'olmo2', 4096, OLMO, 'L'),
    'smollm2-135m':  ('HuggingFaceTB/SmolLM2-135M', None, 'RoPE', 'llama', 8192, APACHE, 'S'),
    'smollm2-360m':  ('HuggingFaceTB/SmolLM2-360M', None, 'RoPE', 'llama', 8192, APACHE, 'S'),
    'smollm2-1.7b':  ('HuggingFaceTB/SmolLM2-1.7B', None, 'RoPE', 'llama', 8192, APACHE, 'S'),
}
ORDER = list(MODELS)
EXCLUDED = {'mosaicml/mpt-7b': 'ALiBi; licence Apache-2.0 would permit, but the hub repo is not accessible '
                               '(HTTP 401 "Repository Not Found" on 2026-10-10), so it cannot be fetched.'}
CLASSES = ('APE', 'ALiBi', 'RoPE')

TASKS = ('copy', 'reverse', 'brackets', 'arith', 'countdown', 'retrieval')
FEWSHOT_TASKS = TASKS[:5]
KS = (16, 64, 256, 1024)          # shifts k of the position indices / pad counts, each only where the context allows
K_PRIMARY = 256                   # shift at which analyze.py checks that the APE models change
WINDOW = 128                      # sliding-window scenario: banded causal mask over the last 128 cache slots
SINK = 4                          # ... plus the first 4 cache slots (attention sinks), see pc_engine.make_mask
SCENARIOS = ('offset', 'leftpad', 'cache', 'window', 'offset_bf16')
FP32_SCEN = SCENARIOS[:4]
RET_TIER = {1024: ('r1k', 700), 2048: ('r2k', 1700)}   # ctx -> (tier, max prompt tokens); ctx >= 4096 -> r4k
RET_TIER_LONG = ('r4k', 3000)
RET_MAX_NEW = 16
CHECK_N = 8                       # check items per task (first 8 of each task in the item file)


def ret_tier(ctx):
    return RET_TIER.get(ctx, RET_TIER_LONG)


def scen_tasks(scen):
    return FEWSHOT_TASKS if scen == 'window' else TASKS   # no retrieval under the sliding window (queried record outside it)


def scen_precision(scen):
    return 'bf16' if scen == 'offset_bf16' else 'fp32'


def units():
    """Global unit list: (uid, model, scenario); index = position in this list."""
    return [(f'{m}__{s}', m, s) for m in ORDER for s in SCENARIOS]


# parameters in billions, given in the cards
PARAMS_B = {'gpt2': 0.124, 'gpt2-medium': 0.355, 'gpt2-large': 0.774, 'gpt2-xl': 1.56, 'gpt-neo-125m': 0.125,
            'gpt-neo-1.3B': 1.37, 'gpt-neo-2.7B': 2.72, 'opt-125m': 0.125, 'opt-350m': 0.331, 'opt-1.3b': 1.32,
            'opt-2.7b': 2.65, 'opt-6.7b': 6.66, 'bloom-560m': 0.559, 'bloom-1b1': 1.07, 'bloom-1b7': 1.72,
            'bloom-3b': 3.0, 'bloom-7b1': 7.07, 'pythia-160m': 0.162, 'pythia-410m': 0.405, 'pythia-1b': 1.01,
            'pythia-1.4b': 1.41, 'pythia-2.8b': 2.78, 'pythia-6.9b': 6.86, 'gpt-j-6b': 6.05, 'qwen2.5-0.5b': 0.494,
            'qwen2.5-1.5b': 1.54, 'qwen2.5-3b': 3.09, 'qwen2.5-7b': 7.62, 'olmo-2-1b': 1.48, 'olmo-2-7b': 7.3,
            'smollm2-135m': 0.135, 'smollm2-360m': 0.362, 'smollm2-1.7b': 1.71}

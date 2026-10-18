"""Cache teacher-forced items for both tokenizer families (CPU, deterministic)."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tf_items import build_items
from common import MODELS, OUT
from transformers import AutoTokenizer
for fam, m in [('gpt2bpe', 'f1'), ('neox', 'py160m')]:
    out = f'{OUT}/results/items_{fam}.json'
    if os.path.exists(out):
        print('exists', out); continue
    tok = AutoTokenizer.from_pretrained(MODELS[m][0])
    items, stats = build_items(tok)
    json.dump(dict(stats=stats, items=items), open(out + '.tmp', 'w'))
    os.replace(out + '.tmp', out)
    print(fam, stats, len(items))

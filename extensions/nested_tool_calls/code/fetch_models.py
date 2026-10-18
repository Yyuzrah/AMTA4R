"""Download the models (safetensors + tokenizer + config + card) into the Hugging Face cache, check that the licence
is permissive (hub tag apache-2.0 or mit) and record licence, revision, local path and architecture in
meta/licenses.json (the folder meta/ must exist). Models already recorded are skipped; writes meta/FETCH_DONE when
every model is present. Usage: python fetch_models.py [model tag ...]"""
import json, os, sys, re, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from huggingface_hub import snapshot_download, HfApi
from models import MODELS, EXCLUDED, PERMISSIVE, ORDER
A = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = f'{A}/meta/licenses.json'
rec = json.load(open(out)) if os.path.exists(out) else {}
api = HfApi()
only = sys.argv[1:] or ORDER
for tag in only:
    hub = MODELS[tag][0]
    if rec.get(tag, {}).get('done') and os.path.exists(rec[tag]['path'] + '/config.json'):
        print('skip', tag, flush=True); continue
    for attempt in range(5):
        try:
            info = api.model_info(hub)
            lic = [t for t in info.tags if t.startswith('license:')]
            assert set(lic) & PERMISSIVE, f'{hub}: licence {lic} not permissive'
            p = snapshot_download(hub, revision=info.sha,
                                  allow_patterns=['*.json', '*.safetensors', 'tokenizer*', 'README.md', '*.model',
                                                  'merges.txt', 'vocab.json', '*.jinja', 'LICENSE*', 'tekken.json'],
                                  ignore_patterns=['consolidated*', 'original/*'])
            break
        except AssertionError:
            raise
        except Exception as ex:
            print('retry', tag, attempt, repr(ex)[:300], flush=True); time.sleep(60)
    else:
        raise SystemExit(f'failed {tag}')
    card = open(f'{p}/README.md').read() if os.path.exists(f'{p}/README.md') else ''
    m = re.search(r'^license:\s*(\S+)', card, re.M)
    cfg = json.load(open(f'{p}/config.json'))
    tc = cfg.get('text_config', cfg)
    rec[tag] = {'hub': hub, 'sha': info.sha, 'license_tag': lic, 'license_card': m.group(1) if m else None,
                'num_hidden_layers': tc.get('num_hidden_layers'), 'hidden_size': tc.get('hidden_size'),
                'num_attention_heads': tc.get('num_attention_heads'), 'intermediate_size': tc.get('intermediate_size'),
                'max_position_embeddings': tc.get('max_position_embeddings'),
                'model_type': cfg.get('model_type'), 'path': p, 'done': True,
                'fetched': time.strftime('%Y-%m-%dT%H:%M:%S')}
    print(tag, {k: v for k, v in rec[tag].items() if k != 'path'}, flush=True)
    json.dump(rec, open(out + '.tmp', 'w'), indent=1); os.replace(out + '.tmp', out)
rec['_excluded'] = EXCLUDED
json.dump(rec, open(out + '.tmp', 'w'), indent=1); os.replace(out + '.tmp', out)
if all(rec.get(t, {}).get('done') for t in ORDER):
    open(f'{A}/meta/FETCH_DONE', 'w').write(time.strftime('%Y-%m-%dT%H:%M:%S') + '\n')
print('ALL DONE', flush=True)

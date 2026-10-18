"""Download one model (safetensors weights, config, tokenizer, model card) into the Hugging Face cache (CPU), check its
license tag against the allowed set, and record the downloaded commit, the architecture and the context length.
Usage: fetch.py <index of the model in registry.ORDER> <base dir>. Writes <base>/meta/fetch/<tag>.json and the empty
file <base>/meta/fetch/fetch_<idx>.done; exits at once if both exist and the snapshot is present."""
import json, os, sys, time, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from registry import MODELS, ORDER

LIC_OK = {'license:mit', 'license:apache-2.0', 'license:other', 'license:bigscience-bloom-rail-1.0'}


def main():
    idx, base = int(sys.argv[1]), sys.argv[2]
    tag = ORDER[idx]
    hub, rev, cls, fam, ctx, lic, size = MODELS[tag]
    out_dir = f'{base}/meta/fetch'
    os.makedirs(out_dir, exist_ok=True)
    mark, rec_p = f'{out_dir}/fetch_{idx}.done', f'{out_dir}/{tag}.json'
    if os.path.exists(mark) and os.path.exists(rec_p):
        r = json.load(open(rec_p))
        if os.path.exists(r['path'] + '/config.json'):
            print('done marker exists', tag); return 0
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    for attempt in range(6):
        try:
            info = api.model_info(hub, revision=rev) if rev else api.model_info(hub)
            tags = [t for t in info.tags if t.startswith('license:')]
            assert set(tags) & LIC_OK, f'{hub}: licence tags {tags} not in the allowed set'
            path = snapshot_download(hub, revision=info.sha,
                                     allow_patterns=['*.json', '*.safetensors', 'tokenizer*', 'README.md', 'merges.txt',
                                                     'vocab.json', '*.model', 'LICENSE*', '*.txt'],
                                     ignore_patterns=['*.msgpack*', '*.h5*', 'onnx/*', 'pytorch_model*', 'tf_model*',
                                                      'flax_model*', 'original/*', 'consolidated*'])
            break
        except AssertionError:
            raise
        except Exception as ex:
            print('retry', tag, attempt, repr(ex)[:300], flush=True)
            time.sleep(60 * (attempt + 1))
    else:
        raise SystemExit(f'fetch failed: {tag}')
    st = [f for f in os.listdir(path) if f.endswith('.safetensors')]
    assert st, f'{tag}: no safetensors in snapshot {path}'
    cfg = json.load(open(f'{path}/config.json'))
    cfg_ctx = cfg.get('max_position_embeddings') or cfg.get('n_positions') or cfg.get('seq_length')
    card = open(f'{path}/README.md').read() if os.path.exists(f'{path}/README.md') else ''
    m = re.search(r'^license:\s*(\S+)', card, re.M)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    probe = tok('Input: a b c', add_special_tokens=True)['input_ids']
    plain = tok('Input: a b c', add_special_tokens=False)['input_ids']
    rec = {'tag': tag, 'hub': hub, 'revision_requested': rev or 'main', 'sha': info.sha, 'path': path,
           'license_tags': tags, 'license_card': m.group(1) if m else None, 'license_registry': lic,
           'class': cls, 'family': fam, 'ctx_registry': ctx, 'ctx_config': cfg_ctx, 'model_type': cfg.get('model_type'),
           'n_layer': cfg.get('num_hidden_layers') or cfg.get('n_layer'),
           'n_head': cfg.get('num_attention_heads') or cfg.get('n_head'),
           'hidden': cfg.get('hidden_size') or cfg.get('n_embd') or cfg.get('d_model'),
           'safetensors': sorted(st), 'tokenizer_len': len(tok),
           'special_tokens_added': len(probe) - len(plain), 'bos_token': tok.bos_token, 'eos_token': tok.eos_token,
           'pad_token': tok.pad_token, 'fetched': time.strftime('%Y-%m-%dT%H:%M:%S')}
    if cfg_ctx is not None:
        assert ctx <= cfg_ctx, f'{tag}: registry ctx {ctx} > config {cfg_ctx}'
    json.dump(rec, open(rec_p + '.tmp', 'w'), indent=1); os.replace(rec_p + '.tmp', rec_p)
    open(mark + '.tmp', 'w').write(f'{tag} {info.sha} {time.strftime("%Y-%m-%dT%H:%M:%S")}\n'); os.replace(mark + '.tmp', mark)
    print(json.dumps({k: v for k, v in rec.items() if k != 'path'}), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

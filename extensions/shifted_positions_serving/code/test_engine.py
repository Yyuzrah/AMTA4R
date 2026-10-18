"""CPU unit tests of the position control in pc_engine.py (analyze.py reads the result as G0): tiny randomly
initialised models of every HF family used (gpt2, gpt_neo, opt, bloom, gpt_neox, gptj, qwen2, olmo2, llama), eager
attention, run in float32 (the production precision) and in float64. Tolerances are relative to max|logit| of the
reference row (PosLM returns float32 logits; HF computes rotary cos/sin in float32 even for a float64 model, so RoPE
translation invariance holds only to ~1e-6 relative; "exact" tests use rel <= 1e-6 in float64 and <= 1e-5 in float32).
In float64 the leftpad tests are skipped: HF's eager softmax casts float64 scores to float32, which turns
finfo(float64).min on the fully masked pad-query rows into -inf -> NaN (an artefact of float64 only; float32/bf16 use
their own finite minimum and are tested in the float32 pass).
T1 explicit position path == default HF forward            (all; exact)
T2 all indices shifted by k: RoPE/ALiBi unchanged (rel <= 1e-4), APE changed (rel >= 1e-3)
T3 gap control (+64 on the second half) changes logits      (all; rel >= 1e-3)
T4 cache: two-phase prefill == single pass (all; exact); block<k> == shift by k (exact); gap<k> changes (>= 1e-3)
T5 leftpad: ref row with P masked pads == no pads (exact); p<k> == shift by k (exact)
T6 window: wide band == full (exact); band 8 != full (>= 1e-3); band with shifted indices unchanged for RoPE/ALiBi
   (<= 1e-4), changed for APE (>= 1e-3)
T7 cached greedy generation == greedy argmax of full recomputation, every row of every scenario (exact tokens)
T8 build_rows bookkeeping (labels, rows, posmeta) for every scenario
T9 prefill in row chunks (1, 2 rows) + merged KV cache + joint decoding == one batch (tokens; first logits exact)
Usage: test_engine.py [out.json]; exit 0 iff every test passes."""
import json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import transformers as tf
import pc_engine as PE

V = 97
FAMS = {
    'gpt2': ('APE', lambda: tf.GPT2Config(vocab_size=V, n_positions=512, n_embd=32, n_layer=2, n_head=4)),
    'gpt_neo': ('APE', lambda: tf.GPTNeoConfig(vocab_size=V, max_position_embeddings=512, hidden_size=32, num_layers=2,
                                               num_heads=4, attention_types=[[['global', 'local'], 1]], window_size=16)),
    'opt': ('APE', lambda: tf.OPTConfig(vocab_size=V, hidden_size=32, num_hidden_layers=2, ffn_dim=64,
                                        num_attention_heads=4, max_position_embeddings=512, word_embed_proj_dim=16,
                                        do_layer_norm_before=False, pad_token_id=1)),
    'bloom': ('ALiBi', lambda: tf.BloomConfig(vocab_size=V, hidden_size=48, n_layer=2, n_head=6)),
    'gpt_neox': ('RoPE', lambda: tf.GPTNeoXConfig(vocab_size=V, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                                                  intermediate_size=64, rotary_pct=0.25, max_position_embeddings=512)),
    'gptj': ('RoPE', lambda: tf.GPTJConfig(vocab_size=V, n_positions=512, n_embd=32, n_layer=2, n_head=4, rotary_dim=4)),
    'qwen2': ('RoPE', lambda: tf.Qwen2Config(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                             num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)),
    'olmo2': ('RoPE', lambda: tf.Olmo2Config(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                             num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)),
    'llama': ('RoPE', lambda: tf.LlamaConfig(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                             num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)),
}
TOL_ROPE, TOL_CHANGE = 1e-4, 1e-3
TOL_EXACT = {'float32': 1e-5, 'float64': 1e-6}


def build(fam, dtype):
    cls, mk = FAMS[fam]
    cfg = mk()
    cfg._attn_implementation = 'eager'
    torch.manual_seed(0)
    m = tf.AutoModelForCausalLM.from_config(cfg).to(getattr(torch, dtype)).eval()
    g = torch.Generator().manual_seed(1)
    for n, p in m.named_parameters():
        is_norm_w = n.endswith('weight') and p.dim() == 1
        p.data = torch.randn(p.shape, generator=g, dtype=p.dtype) * 0.3 + (1.0 if is_norm_w else 0.0)
    return cls, m


def rel(a, b, scale):
    return float((a - b).abs().max() / scale)


def one_row(ids, pos):
    T = len(ids)
    x = torch.tensor(ids)[None]
    return [(x, torch.as_tensor(pos)[None], torch.ones(1, T, dtype=torch.bool))]


def run_family(fam, dtype):
    cls, model = build(fam, dtype)
    EX = TOL_EXACT[dtype]
    do_pad = dtype == 'float32'
    lm = PE.PosLM(model, fam)
    rel_inv = cls in ('RoPE', 'ALiBi')
    res, ok = {}, True
    g = torch.Generator().manual_seed(2)
    T, m, K = 40, 25, [16, 64]
    ids = torch.randint(3, V, (T,), generator=g).tolist()
    ar = torch.arange(T)
    with torch.no_grad():
        plain = PE.full_logits(lm, one_row(ids, ar), None)[0]
        scale = float(plain.abs().max())
        # T1
        d = model(input_ids=torch.tensor(ids)[None]).logits[0, -1].float()
        res['T1'] = rel(d, plain, scale)
        ok &= res['T1'] <= EX
        # T2
        offs = {k: PE.full_logits(lm, one_row(ids, ar + k), None)[0] for k in K}
        res['T2'] = {k: rel(v, plain, scale) for k, v in offs.items()}
        ok &= all((v <= TOL_ROPE) if rel_inv else (v >= TOL_CHANGE) for v in res['T2'].values())
        # T3
        gap = PE.full_logits(lm, one_row(ids, ar + 64 * (ar >= T // 2)), None)[0]
        res['T3'] = rel(gap, plain, scale)
        ok &= res['T3'] >= TOL_CHANGE
        nl = torch.zeros(V, dtype=torch.bool)
        # T4 cache
        labels, phases, win, posmeta = PE.build_rows('cache', ids, m, K, 0, 'cpu')
        gen, first = PE.generate(lm, phases, 1, nl, win)
        L = {l: i for i, l in enumerate(labels)}
        res['T4_ref'] = rel(first[L['ref']], plain, scale)
        res['T4_block'] = {k: rel(first[L[f'block{k}']], offs[k], scale) for k in K}
        res['T4_gap'] = {k: rel(first[L[f'gap{k}']], plain, scale) for k in K}
        res['T4_rep'] = rel(first[L['rep']], first[L['ref']], scale)
        ok &= res['T4_ref'] <= EX and res['T4_rep'] == 0.0
        ok &= all(v <= EX for v in res['T4_block'].values())
        ok &= all(v >= TOL_CHANGE for v in res['T4_gap'].values())
        # T5 leftpad
        if do_pad:
            labels, phases, win, posmeta = PE.build_rows('leftpad', ids, m, K, 0, 'cpu')
            gen, first = PE.generate(lm, phases, 1, nl, win)
            L = {l: i for i, l in enumerate(labels)}
            res['T5_finite'] = bool(torch.isfinite(first).all())
            res['T5_ref'] = rel(first[L['ref']], plain, scale)
            res['T5_pk'] = {k: rel(first[L[f'p{k}']], offs[k], scale) for k in K}
            ok &= res['T5_finite'] and res['T5_ref'] <= EX and all(v <= EX for v in res['T5_pk'].values())
        # T6 window
        wide = PE.full_logits(lm, one_row(ids, ar), None, window=10 ** 6)[0]
        band = PE.full_logits(lm, one_row(ids, ar), None, window=8)[0]
        res['T6_wide'] = rel(wide, plain, scale)
        res['T6_band'] = rel(band, plain, scale)
        bscale = float(band.abs().max())
        res['T6_band_offset'] = {k: rel(PE.full_logits(lm, one_row(ids, ar + k), None, window=8)[0], band, bscale)
                                 for k in K}
        ok &= res['T6_wide'] <= EX and res['T6_band'] >= TOL_CHANGE
        ok &= all((v <= TOL_ROPE) if rel_inv else (v >= TOL_CHANGE) for v in res['T6_band_offset'].values())
        # T7 generation consistency, every scenario, all rows
        res['T7'] = {}
        for scen in (('offset', 'leftpad', 'cache', 'window') if do_pad else ('offset', 'cache', 'window')):
            labels, phases, win, posmeta = PE.build_rows(scen, ids, m, K, 0, 'cpu', window=8)
            gen, first = PE.generate(lm, phases, 6, nl, win)
            agree = True
            for s in range(gen.shape[1]):
                lf = PE.full_logits(lm, phases, gen[:, :s], win)
                agree &= bool((lf.argmax(-1) == gen[:, s]).all())
            res['T7'][scen] = agree
            ok &= agree
            # T9 chunked prefill (1 and 2 rows per chunk) + merged-cache decoding == one batch
            for pc in (1, 2):
                g2, f2 = PE.generate(lm, phases, 6, nl, win, pre_chunk=pc)
                same = bool(g2.shape == gen.shape and (g2 == gen).all()) and rel(f2, first, scale) <= EX
                res.setdefault('T9', {})[f'{scen}/{pc}'] = same
                ok &= same
            # T8 bookkeeping
            R = len(labels)
            assert all(p[0].shape[0] == R and p[1].shape[0] == R for p in phases), scen
            assert len(posmeta) == R and labels[0] == 'ref' and labels[-1] == 'rep', scen
            for lab, (a, b) in zip(labels, posmeta):
                k = int(''.join(c for c in lab if c.isdigit()) or 0)
                exp_first = k if (lab.startswith(('k', 'p', 'block'))) else 0
                assert a == exp_first, (scen, lab, a)
                assert b == T - 1 + (k if not lab.startswith(('ref', 'rep')) else 0), (scen, lab, b)
    res['pass'] = bool(ok)
    return res


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else None
    t0 = time.time()
    allres = {'transformers': tf.__version__, 'torch': torch.__version__, 'families': {}}
    ok = True
    for dtype in ('float32', 'float64'):
        for fam in FAMS:
            try:
                r = run_family(fam, dtype)
            except Exception as ex:
                import traceback; traceback.print_exc()
                r = {'pass': False, 'error': repr(ex)}
            allres['families'][f'{fam}/{dtype}'] = r
            ok &= r['pass']
            print(fam, dtype, FAMS[fam][0], 'PASS' if r['pass'] else 'FAIL',
                  json.dumps({k: v for k, v in r.items() if k != 'pass'}), flush=True)
    allres['pass'] = bool(ok)
    allres['seconds'] = round(time.time() - t0, 1)
    if out:
        json.dump(allres, open(out + '.tmp', 'w'), indent=1); os.replace(out + '.tmp', out)
    print('ALL PASS' if ok else 'SOME FAILED', allres['seconds'], 's')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

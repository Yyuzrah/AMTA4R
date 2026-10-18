"""Lemma 2 at the kernel level on public pretrained language models (App. M.3.1).

For each input sequence x (T tokens) one forward pass with batch B=5 runs the same tokens with position ids
arange(T)+k for k in KS = (0, 0, 16, 64, 256); row 0 is the base, row 1 a replicate control (k=0).
Pre-softmax attention scores A = q k^T * scale (exactly as the model computes them, before masking) are captured
in every layer; statistics are taken over the pairs the layer attends to (causal; GPT-Neo local layers: i-j < 256).
Per sequence we store, per layer: max, sum and rms of |A_0| and the number of attended pairs, and per row b the max
and sum of |A_b - A_0|; per row and position: the top-1 flip against row 0 (all T positions) and the change of
log p(x_{t+1}) (T-1 positions).
--precision fp32 : main run (TF32 off, eager attention, HF code paths unchanged in fp32).
--precision fp64diag : first NDIAG sequences per family (all 128); fp32 forward (scores stored), then fp64 forward of
  a float64 copy with the fp32-only casts lifted (rotary angles, GPT-Neo score cast, softmax dtype, RMSNorm); stores
  the fp64 statistics and the fp32 rounding floor |A_fp32 - A_fp64| per row.
A model whose output file exists is skipped. Output per model: results/raw/<tag>_<precision>.{json,npz}."""
import argparse, json, os, time, copy, math
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
import transformers.models.gpt2.modeling_gpt2 as m_gpt2
import transformers.models.gpt_neo.modeling_gpt_neo as m_neo
import transformers.models.gpt_neox.modeling_gpt_neox as m_neox
import transformers.models.llama.modeling_llama as m_llama

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # extensions/lemma2_pretrained
KS = (0, 0, 16, 64, 256)
FAMILIES = ('countdown_dfs', 'countdown_bfs', 'dyck', 'code')
NDIAG = 128
MODELS = {  # tag: (hub id, encoding, tokenizer key)
    'gpt2': ('gpt2', 'APE', 'gpt2bpe'),
    'gpt2-medium': ('gpt2-medium', 'APE', 'gpt2bpe'),
    'gpt-neo-125m': ('EleutherAI/gpt-neo-125m', 'APE', 'gpt2bpe'),
    'gpt-neo-1.3B': ('EleutherAI/gpt-neo-1.3B', 'APE', 'gpt2bpe'),
    'pythia-160m': ('EleutherAI/pythia-160m', 'RoPE', 'neox'),
    'pythia-410m': ('EleutherAI/pythia-410m', 'RoPE', 'neox'),
    'pythia-1.4b': ('EleutherAI/pythia-1.4b', 'RoPE', 'neox'),
    'smollm2-360m': ('HuggingFaceTB/SmolLM2-360M', 'RoPE', 'smollm2'),
}


class Capture:
    def __init__(self):
        self.on = False
        self.layer_of = {}
        self.reset()

    def reset(self):
        self.recs = {}      # layer -> dict of stats
        self.store = None   # if dict: layer -> stored scores (fp32 reference)
        self.ref = None     # if dict: layer -> fp32 scores to compare against

    def record(self, module, scores, mask):
        """Vectorized: non-attended pairs are zeroed (|.| >= 0, so max and sum over attended pairs are unchanged)."""
        if not self.on:
            return
        li = self.layer_of[id(module)]
        m = mask.to(scores.dtype)
        n = int(mask.sum().item()) * scores.shape[1]
        a0 = scores[0].abs() * m
        r = {'maxA': a0.amax(), 'sumA': a0.sum(dtype=torch.float64), 'n': n,
             'rmsA': (a0.double().pow(2).sum() / n).sqrt()}
        d = (scores - scores[0:1]).abs() * m
        r['maxD'] = d.amax(dim=(1, 2, 3)); r['sumD'] = d.sum(dim=(1, 2, 3), dtype=torch.float64)
        if self.store is not None:
            self.store[li] = scores.detach().clone()
        if self.ref is not None:
            f = (self.ref[li].to(scores.dtype) - scores).abs() * m
            r['maxF'] = f.amax(dim=(1, 2, 3)); r['sumF'] = f.sum(dim=(1, 2, 3), dtype=torch.float64)
        assert li not in self.recs, 'layer recorded twice'
        self.recs[li] = r


CAP = Capture()


def causal(T, device):
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))


# ---- GPT-2: wrap the module-level eager function (called by name at run time) ----
_orig_gpt2 = m_gpt2.eager_attention_forward


def gpt2_eager(module, query, key, value, attention_mask, head_mask=None, **kw):
    if CAP.on:
        s = torch.matmul(query, key.transpose(-1, -2))
        if module.scale_attn_weights:
            s = s / torch.full([], value.size(-1) ** 0.5, dtype=s.dtype, device=s.device)
        if module.scale_attn_by_inverse_layer_idx:
            s = s / float(module.layer_idx + 1)
        T = query.size(-2)
        CAP.record(module, s, module.bias[0, 0, :T, :T].bool())
    return _orig_gpt2(module, query, key, value, attention_mask, head_mask=head_mask, **kw)


m_gpt2.eager_attention_forward = gpt2_eager


# ---- GPT-Neo: re-implementation of _attn, identical in fp32; keeps fp64 in fp64 mode ----
def neo_attn(self, query, key, value, attention_mask=None, head_mask=None):
    dt = torch.float64 if query.dtype == torch.float64 else torch.float32
    query = query.to(dt)
    key = key.to(dt)
    attn_weights = torch.matmul(query, key.transpose(-1, -2))
    query_length, key_length = query.size(-2), key.size(-2)
    causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length]
    CAP.record(self, attn_weights, causal_mask[0, 0].bool())
    mask_value = torch.finfo(attn_weights.dtype).min
    mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
    attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = attn_weights.to(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)
    if head_mask is not None:
        attn_weights = attn_weights * head_mask
    attn_output = torch.matmul(attn_weights, value)
    return attn_output, attn_weights


m_neo.GPTNeoSelfAttention._attn = neo_attn


# ---- GPT-NeoX / Llama: re-implementation of eager_attention_forward (softmax dtype fp32, or fp64 in fp64 mode) ----
def neox_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, head_mask=None, **kw):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    T = query.size(-2)
    CAP.record(module, attn_weights, causal(T, query.device)) if CAP.on else None
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask
    sdt = torch.float64 if query.dtype == torch.float64 else torch.float32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=sdt).to(query.dtype)
    if head_mask is not None:
        attn_weights = attn_weights * head_mask
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def llama_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    key_states = m_llama.repeat_kv(key, module.num_key_value_groups)
    value_states = m_llama.repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    T = query.size(-2)
    CAP.record(module, attn_weights, causal(T, query.device)) if CAP.on else None
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    sdt = torch.float64 if query.dtype == torch.float64 else torch.float32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=sdt).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


m_neox.eager_attention_forward = neox_eager
m_llama.eager_attention_forward = llama_eager


# ---- fp64-only lifts of hard-coded fp32 casts (original code path used for fp32 inputs) ----
def _rot64(orig):
    @torch.no_grad()
    def fwd(self, x, position_ids):
        if x.dtype != torch.float64:
            return orig(self, x, position_ids)
        inv = self.inv_freq[None, :, None].to(torch.float64).expand(position_ids.shape[0], -1, 1).to(x.device)
        pos = position_ids[:, None, :].to(torch.float64)
        freqs = (inv @ pos).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos() * self.attention_scaling, emb.sin() * self.attention_scaling
    return fwd


m_neox.GPTNeoXRotaryEmbedding.forward = _rot64(m_neox.GPTNeoXRotaryEmbedding.forward)
m_llama.LlamaRotaryEmbedding.forward = _rot64(m_llama.LlamaRotaryEmbedding.forward)
_orig_rms = m_llama.LlamaRMSNorm.forward


def rms64(self, hidden_states):
    if hidden_states.dtype != torch.float64:
        return _orig_rms(self, hidden_states)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    return self.weight * (hidden_states * torch.rsqrt(variance + self.variance_epsilon))


m_llama.LlamaRMSNorm.forward = rms64


def attn_modules(model):
    mt = model.config.model_type
    if mt == 'gpt2':
        return [blk.attn for blk in model.transformer.h]
    if mt == 'gpt_neo':
        return [blk.attn.attention for blk in model.transformer.h]
    if mt == 'gpt_neox':
        return [blk.attention for blk in model.gpt_neox.layers]
    if mt == 'llama':
        return [blk.self_attn for blk in model.model.layers]
    raise ValueError(mt)


def load(hub, dev):
    m = AutoModelForCausalLM.from_pretrained(hub, torch_dtype=torch.float32, attn_implementation='eager')
    return m.to(dev).eval()


def register(model):
    for i, mod in enumerate(attn_modules(model)):
        CAP.layer_of[id(mod)] = i


@torch.no_grad()
def forward_rows(model, ids, dev, dtype_logits=None):
    T = len(ids)
    x = torch.tensor(ids, device=dev).unsqueeze(0).expand(len(KS), T).contiguous()
    pos = torch.stack([torch.arange(T, device=dev) + k for k in KS])
    out = model(input_ids=x, position_ids=pos, use_cache=False)
    return x, out.logits


def logit_stats(x, logits):
    lp = torch.log_softmax(logits, dim=-1)
    tgt = x[:, 1:]
    lpt = lp[:, :-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
    top = logits.argmax(-1)  # [B, T]
    flips = (top != top[0:1])  # [B, T]
    dl = lpt - lpt[0:1]
    # base margin top1 - top2 at each position (for flip diagnostics)
    t2 = logits[0].topk(2, dim=-1).values
    margin = (t2[:, 0] - t2[:, 1])
    return {'flips': flips.sum(-1).tolist(), 'flip_pos': flips.cpu().numpy(),
            'sum_dlp': dl.double().sum(-1).tolist(), 'sum_abs_dlp': dl.abs().double().sum(-1).tolist(),
            'dlp_pos': dl.double().cpu().numpy(), 'base_nll_sum': (-lpt[0]).double().sum().item(),
            'flip_margin_max': [margin[flips[b]].max().item() if flips[b].any() else None for b in range(len(KS))],
            'n_pos': int(top.shape[1]), 'n_lp': int(lpt.shape[1])}


def pack(recs, L):
    def st(key):
        return torch.stack([torch.as_tensor(recs[l][key]).double() for l in range(L)]).cpu().numpy()
    o = {'maxA': st('maxA'), 'sumA': st('sumA'), 'rmsA': st('rmsA'), 'n': np.array([recs[l]['n'] for l in range(L)]),
         'maxD': st('maxD'), 'sumD': st('sumD')}
    if 'maxF' in recs[0]:
        o['maxF'] = st('maxF'); o['sumF'] = st('sumF')
    return o


def sanity(model, ids, dev):
    """Control V1 (analyze.py): explicit position ids arange(T) reproduce the default forward pass."""
    CAP.on = False
    x = torch.tensor(ids, device=dev).unsqueeze(0)
    with torch.no_grad():
        a = model(input_ids=x, use_cache=False).logits
        b = model(input_ids=x, position_ids=torch.arange(len(ids), device=dev).unsqueeze(0), use_cache=False).logits
    return (a - b).abs().max().item()


def run_model(tag, inputs, dev, precision, out_path, n_per_family):
    hub, enc, tk = MODELS[tag]
    t0 = time.time()
    model = load(hub, dev)
    register(model)
    L = len(attn_modules(model))
    ctx = getattr(model.config, 'n_positions', None) or model.config.max_position_embeddings
    T = len(inputs[FAMILIES[0]]['tok'][tk][0])
    assert T - 1 + max(KS) < ctx, (T, ctx)
    s1 = sanity(model, inputs[FAMILIES[0]]['tok'][tk][0], dev)
    m64 = None
    if precision == 'fp64diag':
        m64 = copy.deepcopy(model).double().eval()
        register(m64)
    res = {'tag': tag, 'hub': hub, 'encoding': enc, 'tokenizer': tk, 'n_layers': L, 'context': ctx, 'T': T,
           'KS': list(KS), 'precision': precision, 'S1_maxabs_logit_diff_default_vs_arange': s1,
           'device': torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu',
           'torch': torch.__version__, 'families': {}}
    arrays = {}
    for fam in FAMILIES:
        seqs = inputs[fam]['tok'][tk][:n_per_family]
        per = []
        for si, ids in enumerate(seqs):
            if precision == 'fp32':
                CAP.reset(); CAP.on = True
                x, logits = forward_rows(model, ids, dev)
                CAP.on = False
                assert len(CAP.recs) == L, (len(CAP.recs), L)
                rec = pack(CAP.recs, L); rec.update(logit_stats(x, logits))
            else:
                CAP.reset(); CAP.store = {}; CAP.on = True
                x, lg32 = forward_rows(model, ids, dev)
                CAP.on = False
                store = CAP.store
                CAP.reset(); CAP.ref = store; CAP.on = True
                x, lg64 = forward_rows(m64, ids, dev)
                CAP.on = False
                assert len(CAP.recs) == L
                rec = pack(CAP.recs, L); rec.update(logit_stats(x, lg64))
                rec['logit_fp32_vs_fp64_maxabs'] = (lg32.double() - lg64).abs().amax(dim=(1, 2)).tolist()
                del store; CAP.reset()
            per.append(rec)
        keys = ['maxA', 'sumA', 'rmsA', 'n', 'maxD', 'sumD', 'flip_pos', 'dlp_pos'] + (['maxF', 'sumF'] if precision != 'fp32' else [])
        for k in keys:
            arrays[f'{fam}/{k}'] = np.stack([r[k] for r in per])
        res['families'][fam] = {
            'n_seq': len(per), 'item_ids': inputs[fam]['ids'][:n_per_family],
            'flips': [r['flips'] for r in per], 'sum_dlp': [r['sum_dlp'] for r in per],
            'sum_abs_dlp': [r['sum_abs_dlp'] for r in per], 'base_nll_sum': [r['base_nll_sum'] for r in per],
            'flip_margin_max': [r['flip_margin_max'] for r in per], 'n_pos': per[0]['n_pos'], 'n_lp': per[0]['n_lp']}
        if precision != 'fp32':
            res['families'][fam]['logit_fp32_vs_fp64_maxabs'] = [r['logit_fp32_vs_fp64_maxabs'] for r in per]
        print(f'{tag} {precision} {fam} done {len(per)} seqs {time.time()-t0:.0f}s', flush=True)
    res['seconds'] = time.time() - t0
    tmp = out_path + '.tmp'
    np.savez_compressed(out_path.replace('.json', '.npz'), **arrays)
    json.dump(res, open(tmp, 'w'))
    os.replace(tmp, out_path)
    del model, m64
    if dev == 'cuda':
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--precision', choices=['fp32', 'fp64diag'], default='fp32')
    ap.add_argument('--n', type=int, default=None, help='sequences per family (default: all / NDIAG for fp64diag)')
    ap.add_argument('--outdir', default=f'{HERE}/results/raw')
    ap.add_argument('--cpu', action='store_true')
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    dev = 'cpu' if a.cpu or not torch.cuda.is_available() else 'cuda'
    inputs = json.load(open(f'{HERE}/results/inputs.json'))
    n = a.n or (NDIAG if a.precision == 'fp64diag' else inputs['_meta']['N_per_family'])
    os.makedirs(a.outdir, exist_ok=True)
    for tag in a.models:
        out = f'{a.outdir}/{tag}_{a.precision}.json'
        if os.path.exists(out):
            print('skip (exists)', out, flush=True); continue
        run_model(tag, inputs, dev, a.precision, out, n)
        print('wrote', out, flush=True)


if __name__ == '__main__':
    main()

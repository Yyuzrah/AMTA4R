"""Control V3: independent re-implementation of the layer-1 pre-softmax scores from the raw weights
(no HF forward code), in float64 on CPU, for gpt2, gpt-neo-125m, pythia-160m, smollm2-360m and the first 2
sequences of each family. Compares m1_first_max (and the layer-1 score scale) with the captured fp32 values in
results/raw/<tag>_fp32.npz. Needs results/inputs.json and the checkpoints in the Hugging Face cache.
Output: results/verify_layer1.json"""
import json, math, os
import numpy as np
import torch
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # extensions/lemma2_pretrained
KS = (0, 0, 16, 64, 256)
FAM = ('countdown_dfs', 'countdown_bfs', 'dyck', 'code')
NSEQ = 2
D = torch.float64


def ln(h, w, b, eps):
    mu = h.mean(-1, keepdim=True); var = ((h - mu) ** 2).mean(-1, keepdim=True)
    return (h - mu) / torch.sqrt(var + eps) * w + b


def rot_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def rope(x, pos, base, rd):
    # x [H, T, dh]; rotate the first rd dims (NeoX/Llama "rotate_half" convention), angles in float64
    inv = 1.0 / (base ** (torch.arange(0, rd, 2, dtype=D) / rd))
    ang = pos.to(D)[:, None] * inv[None, :]
    emb = torch.cat((ang, ang), -1)
    xr, xp = x[..., :rd], x[..., rd:]
    xr = xr * emb.cos() + rot_half(xr) * emb.sin()
    return torch.cat((xr, xp), -1)


def scores_layer1(tag, W, cfg, ids, k):
    T = len(ids); x = torch.tensor(ids); pos = torch.arange(T) + k
    if tag in ('gpt2', 'gpt-neo-125m'):
        p = '' if tag == 'gpt2' else 'transformer.'
        h = W[p + 'wte.weight'][x] + W[p + 'wpe.weight'][pos]
        h = ln(h, W[p + 'h.0.ln_1.weight'], W[p + 'h.0.ln_1.bias'], cfg['layer_norm_epsilon'])
        H = cfg.get('n_head') or cfg['num_heads']; E = h.shape[-1]; dh = E // H
        if tag == 'gpt2':
            qkv = h @ W['h.0.attn.c_attn.weight'] + W['h.0.attn.c_attn.bias']
            q, kk = qkv[:, :E], qkv[:, E:2 * E]
            scale = 1 / math.sqrt(dh)
        else:
            q = h @ W[p + 'h.0.attn.attention.q_proj.weight'].T
            kk = h @ W[p + 'h.0.attn.attention.k_proj.weight'].T
            scale = 1.0
        q = q.view(T, H, dh).transpose(0, 1); kk = kk.view(T, H, dh).transpose(0, 1)
        return (q @ kk.transpose(-1, -2)) * scale
    if tag == 'pythia-160m':
        h = W['gpt_neox.embed_in.weight'][x]
        h = ln(h, W['gpt_neox.layers.0.input_layernorm.weight'], W['gpt_neox.layers.0.input_layernorm.bias'], cfg['layer_norm_eps'])
        H = cfg['num_attention_heads']; dh = h.shape[-1] // H
        qkv = h @ W['gpt_neox.layers.0.attention.query_key_value.weight'].T + W['gpt_neox.layers.0.attention.query_key_value.bias']
        qkv = qkv.view(T, H, 3 * dh).transpose(0, 1)
        q, kk = qkv[..., :dh], qkv[..., dh:2 * dh]
        rd = int(dh * cfg['rotary_pct'])
        q = rope(q, pos, cfg['rotary_emb_base'], rd); kk = rope(kk, pos, cfg['rotary_emb_base'], rd)
        return (q @ kk.transpose(-1, -2)) / math.sqrt(dh)
    if tag == 'smollm2-360m':
        h = W['model.embed_tokens.weight'][x]
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + cfg['rms_norm_eps']) * W['model.layers.0.input_layernorm.weight']
        H = cfg['num_attention_heads']; Hk = cfg['num_key_value_heads']; dh = cfg['hidden_size'] // H
        q = (h @ W['model.layers.0.self_attn.q_proj.weight'].T).view(T, H, dh).transpose(0, 1)
        kk = (h @ W['model.layers.0.self_attn.k_proj.weight'].T).view(T, Hk, dh).transpose(0, 1)
        kk = kk.repeat_interleave(H // Hk, dim=0)
        q = rope(q, pos, cfg['rope_theta'], dh); kk = rope(kk, pos, cfg['rope_theta'], dh)
        return (q @ kk.transpose(-1, -2)) / math.sqrt(dh)
    raise ValueError(tag)


HUB = {'gpt2': 'gpt2', 'gpt-neo-125m': 'EleutherAI/gpt-neo-125m', 'pythia-160m': 'EleutherAI/pythia-160m',
       'smollm2-360m': 'HuggingFaceTB/SmolLM2-360M'}
TOK = {'gpt2': 'gpt2bpe', 'gpt-neo-125m': 'gpt2bpe', 'pythia-160m': 'neox', 'smollm2-360m': 'smollm2'}


def main():
    inputs = json.load(open(f'{HERE}/results/inputs.json'))
    out = {}
    for tag, hub in HUB.items():
        d = snapshot_download(hub, local_files_only=True)
        cfg = json.load(open(f'{d}/config.json'))
        W = {k: v.to(D) for k, v in load_file(f'{d}/model.safetensors').items()}
        if tag == 'gpt2' and 'h.0.ln_1.weight' not in W:
            W = {k.replace('transformer.', ''): v for k, v in W.items()}
        z = np.load(f'{HERE}/results/raw/{tag}_fp32.npz')
        mx = np.zeros(len(KS)); maxA = 0.0; sumD = np.zeros(len(KS)); sumA = 0.0
        cap_maxD = np.zeros(len(KS)); cap_maxA = 0.0; cap_sumD = np.zeros(len(KS)); cap_sumA = 0.0
        for fam in FAM:
            for si in range(NSEQ):
                ids = inputs[fam]['tok'][TOK[tag]][si]
                T = len(ids); mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
                A0 = scores_layer1(tag, W, cfg, ids, 0)
                sel = mask.expand(A0.shape)
                maxA = max(maxA, A0[sel].abs().max().item()); sumA += A0[sel].abs().sum().item()
                for b, k in enumerate(KS):
                    dd = (scores_layer1(tag, W, cfg, ids, k) - A0)[sel].abs()
                    mx[b] = max(mx[b], dd.max().item()); sumD[b] += dd.sum().item()
                cap_maxD = np.maximum(cap_maxD, z[f'{fam}/maxD'][si, 0]); cap_maxA = max(cap_maxA, z[f'{fam}/maxA'][si, 0])
                cap_sumD += z[f'{fam}/sumD'][si, 0]; cap_sumA += z[f'{fam}/sumA'][si, 0]
        ind = (mx / maxA).tolist(); cap = (cap_maxD / cap_maxA).tolist()
        out[tag] = {'independent_fp64_m1_first_max': dict(zip(map(str, KS), ind)),
                    'captured_fp32_m1_first_max': dict(zip(map(str, KS), cap)),
                    'independent_fp64_m1_first_mean': dict(zip(map(str, KS), (sumD / sumA).tolist())),
                    'captured_fp32_m1_first_mean': dict(zip(map(str, KS), (cap_sumD / cap_sumA).tolist())),
                    'scale_maxA_rel_diff': abs(maxA - cap_maxA) / maxA, 'scale_sumA_rel_diff': abs(sumA - cap_sumA) / sumA,
                    'n_seq': NSEQ * len(FAM)}
        print(tag, json.dumps(out[tag]), flush=True)
    json.dump(out, open(f'{HERE}/results/verify_layer1.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

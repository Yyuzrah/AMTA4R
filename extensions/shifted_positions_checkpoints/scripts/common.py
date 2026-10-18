"""Model registry, model loading, batched greedy continuation with explicit position ids, layer-0 score capture."""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repository root
OUT = os.environ.get('SHIFT_OUT', f'{ROOT}/outputs/shifted_positions_checkpoints')
DATA = os.environ.get('SHIFT_DATA', f'{ROOT}/data/experiment_500k')
DEV = os.environ.get('SHIFT_DEVICE', 'cuda')
MODELS = {
    # name: (path, pe, family)
    'f1': (os.environ.get('CKPT_F1', f'{ROOT}/checkpoints/full/small_deep_mixed_rope/final_model'), 'rope', 'neox'),
    'f2': (os.environ.get('CKPT_F2', f'{ROOT}/checkpoints/full/large_deep_bfs_ape/final_model'), 'ape', 'neo'),
    'neo125m': ('EleutherAI/gpt-neo-125m', 'ape', 'neo'),
    'py160m': ('EleutherAI/pythia-160m', 'rope', 'neox'),
    'neo1.3b': ('EleutherAI/gpt-neo-1.3B', 'ape', 'neo'),
    'py1.4b': ('EleutherAI/pythia-1.4b', 'rope', 'neox'),
    'neo2.7b': ('EleutherAI/gpt-neo-2.7B', 'ape', 'neo'),
    'py2.8b': ('EleutherAI/pythia-2.8b', 'rope', 'neox'),
}


def load(name, device=None):
    device = device or DEV
    path, pe, fam = MODELS[name]
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, attn_implementation='eager').to(device).eval()
    return tok, model, pe, fam


def bos_id(tok):
    return tok.convert_tokens_to_ids('<|endoftext|>')


@torch.no_grad()
def route_logits(model, ids, pos, gather):
    """ids,pos: LongTensor [B,T]; a position < 0 marks a padding token (masked out, its position set to 0); gather: list
    (per row) of positions whose next-token logits are needed. Returns list of [n_i, V] float tensors."""
    B, T = ids.shape
    lens = [len(g) for g in gather]
    att = (pos >= 0).long(); pos = pos.clamp(min=0)
    h = model.base_model(input_ids=ids, attention_mask=att, position_ids=pos, use_cache=False).last_hidden_state
    out = []
    head = model.get_output_embeddings()
    for b in range(B):
        g = torch.tensor(gather[b], device=ids.device)
        out.append(head(h[b, g]).float())
    return out


@torch.no_grad()
def greedy(model, rows, eos, max_bs=64, device=None):
    """rows: list of dict(ids=list, pos=list, budget=int). Greedy decoding, KV cache, left padding, explicit positions.
    Each row stops after `budget` new tokens or after emitting eos (eos included). Returns list of token lists."""
    device = device or DEV
    order = sorted(range(len(rows)), key=lambda i: (len(rows[i]['ids']), rows[i]['budget']))
    res = [None] * len(rows)
    for s in range(0, len(order), max_bs):
        idx = order[s:s + max_bs]
        B = len(idx); P = max(len(rows[i]['ids']) for i in idx)
        ids = torch.full((B, P), eos, dtype=torch.long); att = torch.zeros((B, P), dtype=torch.long)
        pos = torch.zeros((B, P), dtype=torch.long)
        for r, i in enumerate(idx):
            n = len(rows[i]['ids']); off = P - n
            ids[r, off:] = torch.tensor(rows[i]['ids']); att[r, off:] = 1; pos[r, off:] = torch.tensor(rows[i]['pos'])
        ids, att, pos = ids.to(device), att.to(device), pos.to(device)
        budget = torch.tensor([rows[i]['budget'] for i in idx], device=device)
        last = pos[:, -1].clone()
        out = model(input_ids=ids, attention_mask=att, position_ids=pos, use_cache=True)
        past = out.past_key_values; nxt = out.logits[:, -1].argmax(-1)
        gen = []; done = budget <= 0
        maxb = int(budget.max().item())
        for step in range(maxb):
            gen.append(torch.where(done, torch.full_like(nxt, -1), nxt))
            done = done | (nxt == eos) | (step + 1 >= budget)
            if bool(done.all()):
                break
            att = torch.cat([att, torch.ones((B, 1), dtype=torch.long, device=device)], 1)
            last = last + 1
            out = model(input_ids=nxt[:, None], attention_mask=att, position_ids=last[:, None], past_key_values=past, use_cache=True)
            past = out.past_key_values; nxt = out.logits[:, -1].argmax(-1)
        G = torch.stack(gen, 1).tolist() if gen else [[] for _ in idx]
        for r, i in enumerate(idx):
            res[i] = [t for t in G[r] if t >= 0]
        del past, out
    return res


# ---- capture of layer-0 pre-softmax attention scores (Lemma 2 check) ----
CAPTURE = {'on': False, 'scores': None}


def install_capture():
    import transformers.models.gpt_neox.modeling_gpt_neox as mx
    import transformers.models.gpt_neo.modeling_gpt_neo as mn
    if getattr(mx, '_capture_patched', False):
        return
    orig_eager = mx.eager_attention_forward

    def eager_wrap(module, query, key, value, attention_mask, scaling, **kw):
        if CAPTURE['on'] and getattr(module, 'layer_idx', None) == 0:
            CAPTURE['scores'] = (torch.matmul(query, key.transpose(2, 3)) * scaling).detach().float().cpu()
        return orig_eager(module, query, key, value, attention_mask, scaling, **kw)
    mx.eager_attention_forward = eager_wrap
    orig_attn = mn.GPTNeoSelfAttention._attn

    def attn_wrap(self, query, key, value, attention_mask=None, head_mask=None):
        if CAPTURE['on'] and self.layer_id == 0:
            CAPTURE['scores'] = torch.matmul(query.float(), key.float().transpose(-1, -2)).detach().cpu()
        return orig_attn(self, query, key, value, attention_mask, head_mask)
    mn.GPTNeoSelfAttention._attn = attn_wrap
    mx._capture_patched = True

"""Position control + greedy generation for every model family of the registry.
- Position ids are passed explicitly (`position_ids`) for APE and RoPE families (OPT adds 2 to the ids internally to
  index its position table, so the ids here are the logical 0-based positions).
- BLOOM (ALiBi) has no position ids: HF builds the bias slope_h * position(key) from the attention mask (softmax is
  invariant to the per-query constant -slope_h * position(query)). The instance method `build_alibi_tensor` is
  replaced by one that uses the explicit key positions of the current call; the slopes are taken from HF's own
  build_alibi_tensor on the same device, so at positions 0..T-1 the bias is bit-identical to HF's (pc_run.py compares
  the two paths on every unit, key G1_default_vs_explicit).
- Attention mask: 2-D (1 = attend) for full attention (always passed, so HF never infers packed sequences from
  position ids); 4-D additive (0 / dtype-min) banded causal mask over cache indices for the sliding-window scenario
  (window W = 128 most recent cache slots + SINK = 4 first slots).
- Logits are computed only at the positions needed (base model + output head), never for the whole prompt.
- Row order inside one item: 'ref' first, perturbed rows, the replicate 'rep' (= ref) LAST, so that when the rows of
  an item are split into chunks the replicate measures the cross-chunk numerical noise floor."""
import torch

WINDOW_DEFAULT = 128
SINK = 4          # sliding-window scenario: StreamingLLM-style attention sinks (the first 4 cache slots stay visible)


class PosLM:
    def __init__(self, model, family):
        self.model, self.family = model, family
        self.base = model.base_model
        self.head = model.get_output_embeddings()
        self.dtype = next(model.parameters()).dtype
        cfg = model.config
        self.n_head = cfg.num_attention_heads
        self.n_layer = cfg.num_hidden_layers
        self.n_kv = getattr(cfg, 'num_key_value_heads', None) or self.n_head
        self.hidden = cfg.hidden_size
        self.head_dim = getattr(cfg, 'head_dim', None) or self.hidden // self.n_head
        self.inter = getattr(cfg, 'intermediate_size', None) or getattr(cfg, 'n_inner', None) or \
            getattr(cfg, 'ffn_dim', None) or 4 * self.hidden
        self.keypos = None
        self._slopes = {}
        if family == 'bloom':
            orig = self.base.build_alibi_tensor

            def alibi(attention_mask, num_heads, dtype):
                kp = self.keypos
                if kp is None:
                    return orig(attention_mask, num_heads, dtype)
                dev = kp.device
                if dev not in self._slopes:   # slopes exactly as HF computes them, on the same device
                    probe = orig(torch.ones(1, 2, dtype=torch.long, device=dev), num_heads, torch.float32)
                    self._slopes[dev] = probe.reshape(num_heads, 2)[:, 1].contiguous()
                B, kv = kp.shape
                al = self._slopes[dev][None, :, None] * kp[:, None, :]      # fp32, same arithmetic as HF
                return al.reshape(B * num_heads, 1, kv).to(dtype)
            self.base.build_alibi_tensor = alibi

    def __call__(self, ids, pos, keypos, mask, past=None, last_n=1, use_cache=True):
        kw = dict(input_ids=ids, attention_mask=mask, past_key_values=past, use_cache=use_cache)
        if self.family == 'bloom':
            self.keypos = keypos
        else:
            kw['position_ids'] = pos
        try:
            out = self.base(**kw)
        finally:
            self.keypos = None
        h = out.last_hidden_state[:, -last_n:]
        lg = self.head(h).float()
        return (lg[:, -1] if last_n == 1 else lg), (out.past_key_values if use_cache else None)

    def cache_bytes(self, kv_final, elt):
        """KV cache bytes per row at the final length (+ 50% for the per-step concatenation copies)."""
        return int(1.5 * 2 * self.n_layer * self.n_kv * self.head_dim * kv_final * elt)

    def row_bytes(self, q, kv_final, elt):
        """Rough peak bytes per batch row in a prefill: attention scores (x3: scores, masked scores, softmax; fp32),
        4-D mask, KV cache, MLP activations."""
        attn = 3 * self.n_head * q * kv_final * 4 + q * kv_final * 4
        cache = 2 * self.n_layer * self.n_kv * self.head_dim * kv_final * elt
        mlp = 3 * q * self.inter * max(elt, 4) + 4 * q * self.hidden * max(elt, 4)
        return attn + cache + mlp


def make_mask(valid, kv0, q, window, dtype, sink=SINK):
    """valid: (R, kv) bool over all cache slots incl. the q new ones (last q). Returns a 2-D long mask (window None)
    or a 4-D additive banded causal mask (R, 1, q, kv) over cache indices: query i attends keys j <= i with
    i - window < j (the window) or j < sink (attention sinks, as in StreamingLLM-style sliding-window serving)."""
    if window is None:
        return valid.long()
    R, kv = valid.shape
    dev = valid.device
    qi = torch.arange(kv0, kv0 + q, device=dev)[:, None]
    kj = torch.arange(kv, device=dev)[None, :]
    allowed = (kj <= qi) & ((kj > qi - window) | (kj < sink))
    allowed = allowed[None] & valid[:, None, :]
    m = torch.zeros(R, 1, q, kv, dtype=dtype, device=dev)
    return m.masked_fill(~allowed[:, None], torch.finfo(dtype).min)


def build_rows(scen, ids, m, K, pad_id, dev, window=WINDOW_DEFAULT):
    """Rows of one item for a scenario. Returns (labels, phases, window, posmeta); phases = [(ids, pos, valid)]
    run in order with the KV cache; posmeta = per row (first real position, last prompt position).
    offset / offset_bf16 / window: rows ref (0..T-1), k<k> (k..k+T-1), rep.
    leftpad: P = max(K) masked pad tokens in front of every row; ref/rep take positions from the mask (real tokens
      0..T-1); p<k> = naive absolute ids over a k-pad prefix (real tokens k..k+T-1).
    cache: prompt prefilled in two phases (prefix [0, m) into the KV cache, then the suffix); ref (0..T-1);
      block<k> = whole prompt translated by k (cache computed at shifted positions, all indices shifted);
      gap<k> = cached prefix at 0..m-1, suffix placed at m+k.. (relative distances across the cache boundary change)."""
    T = len(ids)
    x = torch.tensor(ids, device=dev, dtype=torch.long)
    ar = torch.arange(T, device=dev)
    win = None
    if scen in ('offset', 'offset_bf16', 'window'):
        labels = ['ref'] + [f'k{k}' for k in K] + ['rep']
        offs = [0] + list(K) + [0]
        R = len(offs)
        pos = torch.stack([ar + o for o in offs])
        phases = [(x[None].expand(R, T), pos, torch.ones(R, T, dtype=torch.bool, device=dev))]
        posmeta = [(o, o + T - 1) for o in offs]
        win = window if scen == 'window' else None
    elif scen == 'leftpad':
        P = max(K) if K else 0
        labels = ['ref'] + [f'p{k}' for k in K] + ['rep']
        R = len(labels)
        xx = torch.cat([torch.full((P,), pad_id, device=dev, dtype=torch.long), x])[None].expand(R, P + T)
        valid = torch.cat([torch.zeros(P, dtype=torch.bool, device=dev),
                           torch.ones(T, dtype=torch.bool, device=dev)])[None].expand(R, P + T)
        full = torch.arange(P + T, device=dev)
        mrow = torch.cat([torch.zeros(P, dtype=torch.long, device=dev), ar])           # positions from the mask
        rows = [mrow] + [(full - (P - k)).clamp(min=0) for k in K] + [mrow]           # absolute ids, k pads
        phases = [(xx, torch.stack(rows), valid)]
        posmeta = [(0, T - 1)] + [(k, k + T - 1) for k in K] + [(0, T - 1)]
    elif scen == 'cache':
        labels = ['ref'] + [f'block{k}' for k in K] + [f'gap{k}' for k in K] + ['rep']
        pre = [0] + list(K) + [0] * len(K) + [0]
        suf = [0] + list(K) + list(K) + [0]
        R = len(labels)
        ones = lambda n: torch.ones(R, n, dtype=torch.bool, device=dev)
        phases = [(x[:m][None].expand(R, m), torch.stack([ar[:m] + o for o in pre]), ones(m)),
                  (x[m:][None].expand(R, T - m), torch.stack([ar[m:] + o for o in suf]), ones(T - m))]
        posmeta = [(a, b + T - 1) for a, b in zip(pre, suf)]
    else:
        raise ValueError(scen)
    return labels, phases, win, posmeta


def slice_phases(phases, rows):
    return [(i[rows], p[rows], v[rows]) for i, p, v in phases]


def merge_caches(caches):
    """Concatenate the KV caches of row chunks (same token count) along the batch dimension, layer by layer."""
    c0 = caches[0]
    if len(caches) == 1:
        return c0
    for i, layer in enumerate(c0.layers):           # layer by layer, releasing the chunk tensors (peak +1 layer)
        layer.keys = torch.cat([c.layers[i].keys for c in caches], 0)
        layer.values = torch.cat([c.layers[i].values for c in caches], 0)
        for c in caches[1:]:
            c.layers[i].keys = c.layers[i].values = None
    return c0


@torch.no_grad()
def generate(lm, phases, max_new, nl, window=None, pre_chunk=None):
    """Prefill the phases with the KV cache (in row chunks of `pre_chunk` rows if given: the prefill attention
    memory grows with q x kv per row, decoding only with the cache), then greedy-decode all rows of the call together
    up to max_new tokens (stop when every row emitted a stop token). Positions continue from each row's last prompt
    position. Returns gen (R, S) long, first-step logits (R, V) float32."""
    R, dev = phases[0][0].shape[0], phases[0][0].device
    keypos = torch.cat([p[1] for p in phases], 1)
    valid = torch.cat([p[2] for p in phases], 1)
    c = R if pre_chunk is None else max(1, min(R, pre_chunk))
    caches, firsts = [], []
    for s in range(0, R, c):
        rows = slice(s, min(R, s + c))
        past, kv0 = None, 0
        for ids, pos, val in phases:
            q = ids.shape[1]
            logits, past = lm(ids[rows], pos[rows], keypos[rows, :kv0 + q],
                              make_mask(valid[rows, :kv0 + q], kv0, q, window, lm.dtype), past)
            kv0 += q
        caches.append(past); firsts.append(logits)
    past = merge_caches(caches)
    logits = torch.cat(firsts, 0)
    first = logits
    last = phases[-1][1][:, -1]
    gen, done = [], torch.zeros(R, dtype=torch.bool, device=dev)
    nxt = logits.argmax(-1)
    one = torch.ones(R, 1, dtype=torch.bool, device=dev)
    for s in range(max_new):
        gen.append(nxt)
        done |= nl[nxt]
        if bool(done.all()) or s == max_new - 1:
            break
        p = (last + 1 + s)[:, None]
        kv0 = keypos.shape[1]
        keypos = torch.cat([keypos, p], 1)
        valid = torch.cat([valid, one], 1)
        logits, past = lm(nxt[:, None], p, keypos, make_mask(valid, kv0, 1, window, lm.dtype), past)
        nxt = logits.argmax(-1)
    return torch.stack(gen, 1), first


@torch.no_grad()
def full_logits(lm, phases, extra_tokens, window=None, last_n=1):
    """No-cache forward over all phase tokens plus `extra_tokens` (R, g) appended with continuing positions.
    Returns logits at the last `last_n` positions, (R, last_n, V) or (R, V) if last_n == 1."""
    ids = torch.cat([p[0] for p in phases], 1)
    pos = torch.cat([p[1] for p in phases], 1)
    val = torch.cat([p[2] for p in phases], 1)
    if extra_tokens is not None and extra_tokens.shape[1] > 0:
        g = extra_tokens.shape[1]
        last = phases[-1][1][:, -1:]
        ids = torch.cat([ids, extra_tokens], 1)
        pos = torch.cat([pos, last + 1 + torch.arange(g, device=ids.device)[None]], 1)
        val = torch.cat([val, torch.ones_like(extra_tokens, dtype=torch.bool)], 1)
    lg, _ = lm(ids, pos, pos, make_mask(val, 0, ids.shape[1], window, lm.dtype), None, last_n=last_n,
               use_cache=False)
    return lg


def newline_mask(tok, V, dev):
    m = torch.zeros(V, dtype=torch.bool)
    n = len(tok)
    for i in range(min(V, n)):
        try:
            if '\n' in tok.decode([i]):
                m[i] = True
        except Exception:
            pass
    return m.to(dev)


def stop_mask(tok, V, dev):
    """Greedy decoding stops a row at a token containing a newline or at the EOS token."""
    m = newline_mask(tok, V, dev)
    if tok.eos_token_id is not None and tok.eos_token_id < V:
        m[tok.eos_token_id] = True
    return m


def answer(tok, g, ntok=None):
    """Answer string: generation up to (excluding) the first EOS token, decoded, up to the first newline, stripped.
    Ids beyond the tokenizer's vocabulary (padded embedding rows) are dropped. `ntok` = len(tok), passed in because
    len() of a fast tokenizer rebuilds the vocabulary (slow for 150k-250k vocabularies)."""
    eos = tok.eos_token_id
    if eos is not None and eos in g:
        g = g[:g.index(eos)]
    n = len(tok) if ntok is None else ntok
    g = [t for t in g if t < n]
    return tok.decode(g).split('\n')[0].strip()


def first_stats(first):
    """first-step logits (R, V) float32 -> argmax, max-abs diff vs row 0, KL(p_row0 || p_row)."""
    lp = torch.log_softmax(first.double(), -1)
    kl = (lp[0:1].exp() * (lp[0:1] - lp)).sum(-1)
    return {'argmax': first.argmax(-1).tolist(), 'maxdiff': (first - first[0:1]).abs().amax(-1).tolist(),
            'kl': kl.clamp(min=0).tolist()}

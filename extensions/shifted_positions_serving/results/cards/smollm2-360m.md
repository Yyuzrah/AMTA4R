# Answers under shifted position indices: smollm2-360m

- Hugging Face Hub model `HuggingFaceTB/SmolLM2-360M`; license: Apache-2.0
- position class **RoPE** (llama), trained context 8192, 0.362 B params

| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max abs logit diff |
|---|---|---|---|---|---|---|---|
| offset | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.483 -> 0.483 | 0.0000 | 0.0000 | 0 |
| offset | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.483 -> 0.483 | 0.0000 | 0.0000 | 0 |
| offset | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.483 -> 0.483 | 0.0000 | 0.0000 | 0 |
| offset | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.483 -> 0.483 | 0.0000 | 0.0000 | 0.0001 |
| leftpad | p16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.483 -> 0.483 | 0.0000 | 0.0000 | 0 |
| leftpad | p64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.483 -> 0.483 | 0.0000 | 0.0000 | 0 |
| leftpad | p256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.483 -> 0.483 | 0.0000 | 0.0000 | 0 |
| leftpad | p1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.483 -> 0.483 | 0.0000 | 0.0000 | 0.0001 |
| cache | block16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.483 -> 0.483 | 0.0000 | 0.0000 | 0 |
| cache | block64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.483 -> 0.483 | 0.0000 | 0.0000 | 0 |
| cache | block256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.483 -> 0.483 | 0.0000 | 0.0000 | 0 |
| cache | block1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.483 -> 0.483 | 0.0000 | 0.0000 | 0.0001 |
| cache | gap16 | degrades | -0.1340 [-0.1477, -0.1207] | 0.483 -> 0.349 | 0.4583 | 0.1467 | 3.52 |
| cache | gap64 | degrades | -0.1353 [-0.1500, -0.1207] | 0.483 -> 0.348 | 0.5243 | 0.1673 | 4.15 |
| cache | gap256 | degrades | -0.1933 [-0.2083, -0.1777] | 0.483 -> 0.290 | 0.6533 | 0.2020 | 4.12 |
| cache | gap1024 | degrades | -0.2257 [-0.2410, -0.2100] | 0.483 -> 0.257 | 0.6950 | 0.2367 | 5.52 |
| window | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.546 -> 0.546 | 0.0000 | 0.0000 | 0 |
| window | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.546 -> 0.546 | 0.0000 | 0.0000 | 0 |
| window | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.546 -> 0.546 | 0.0000 | 0.0000 | 0 |
| window | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.546 -> 0.546 | 0.0000 | 0.0000 | 0.0001 |
| offset_bf16 | k16 | unchanged | -0.0027 [-0.0070, +0.0017] | 0.479 -> 0.476 | 0.0577 | 0.0220 | 0.274 |
| offset_bf16 | k64 | unchanged | +0.0010 [-0.0037, +0.0057] | 0.479 -> 0.480 | 0.0580 | 0.0187 | 0.295 |
| offset_bf16 | k256 | unchanged | -0.0027 [-0.0073, +0.0020] | 0.479 -> 0.476 | 0.0600 | 0.0207 | 0.301 |
| offset_bf16 | k1024 | unchanged | -0.0003 [-0.0053, +0.0047] | 0.479 -> 0.479 | 0.0663 | 0.0240 | 0.312 |

**Serving recommendations (descriptive)**

- shift all position ids by a constant: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- left-pad with naive absolute position ids: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- reuse a prefix KV cache computed with all position indices shifted (block): accuracy unchanged at every tested k (max answer-flip rate 0.000)
- append to a cached prefix with a position gap: accuracy changes at gap16, gap64, gap256, gap1024 -> recompute/reset positions
- start a 128-token sliding window at a non-zero position: accuracy unchanged at every tested k (max answer-flip rate 0.000)

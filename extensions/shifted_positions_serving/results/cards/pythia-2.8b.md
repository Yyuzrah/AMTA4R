# Answers under shifted position indices: pythia-2.8b

- Hugging Face Hub model `EleutherAI/pythia-2.8b`; license: Apache-2.0
- position class **RoPE** (gpt_neox), trained context 2048, 2.78 B params

| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max abs logit diff |
|---|---|---|---|---|---|---|---|
| offset | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.573 -> 0.573 | 0.0000 | 0.0000 | 0.0003 |
| offset | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.573 -> 0.573 | 0.0000 | 0.0000 | 0.0003 |
| offset | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.573 -> 0.573 | 0.0000 | 0.0000 | 0.0003 |
| offset | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.646 -> 0.646 | 0.0000 | 0.0000 | 0.0003 |
| leftpad | p16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.573 -> 0.573 | 0.0000 | 0.0000 | 0.0003 |
| leftpad | p64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.573 -> 0.573 | 0.0000 | 0.0000 | 0.0003 |
| leftpad | p256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.573 -> 0.573 | 0.0000 | 0.0000 | 0.0003 |
| leftpad | p1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.646 -> 0.646 | 0.0000 | 0.0000 | 0.0003 |
| cache | block16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.573 -> 0.573 | 0.0000 | 0.0000 | 0.0003 |
| cache | block64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.573 -> 0.573 | 0.0000 | 0.0000 | 0.0003 |
| cache | block256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.573 -> 0.573 | 0.0000 | 0.0000 | 0.0003 |
| cache | block1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.646 -> 0.646 | 0.0000 | 0.0000 | 0.0003 |
| cache | gap16 | degrades | -0.1417 [-0.1553, -0.1283] | 0.573 -> 0.432 | 0.4597 | 0.2933 | 2.59 |
| cache | gap64 | degrades | -0.1543 [-0.1700, -0.1390] | 0.573 -> 0.419 | 0.5187 | 0.2990 | 3.35 |
| cache | gap256 | degrades | -0.1937 [-0.2100, -0.1773] | 0.573 -> 0.380 | 0.6243 | 0.3760 | 4.06 |
| cache | gap1024 | degrades | -0.3352 [-0.3512, -0.3192] | 0.646 -> 0.311 | 0.6804 | 0.4372 | 7.09 |
| window | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.627 -> 0.627 | 0.0000 | 0.0000 | 0.0003 |
| window | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.627 -> 0.627 | 0.0000 | 0.0000 | 0.0003 |
| window | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.627 -> 0.627 | 0.0000 | 0.0000 | 0.0003 |
| window | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.627 -> 0.627 | 0.0000 | 0.0000 | 0.0003 |
| offset_bf16 | k16 | unchanged | -0.0023 [-0.0107, +0.0060] | 0.573 -> 0.571 | 0.1933 | 0.1317 | 2.06 |
| offset_bf16 | k64 | unchanged | -0.0037 [-0.0117, +0.0043] | 0.573 -> 0.569 | 0.1900 | 0.1253 | 2.06 |
| offset_bf16 | k256 | unchanged | -0.0003 [-0.0083, +0.0077] | 0.573 -> 0.573 | 0.1853 | 0.1227 | 2.06 |
| offset_bf16 | k1024 | unchanged | -0.0036 [-0.0116, +0.0044] | 0.648 -> 0.644 | 0.1208 | 0.0488 | 2.05 |

**Serving recommendations (descriptive)**

- shift all position ids by a constant: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- left-pad with naive absolute position ids: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- reuse a prefix KV cache computed with all position indices shifted (block): accuracy unchanged at every tested k (max answer-flip rate 0.000)
- append to a cached prefix with a position gap: accuracy changes at gap16, gap64, gap256, gap1024 -> recompute/reset positions
- start a 128-token sliding window at a non-zero position: accuracy unchanged at every tested k (max answer-flip rate 0.000)

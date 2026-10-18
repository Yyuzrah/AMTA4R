# Answers under shifted position indices: pythia-1.4b

- Hugging Face Hub model `EleutherAI/pythia-1.4b`; license: Apache-2.0
- position class **RoPE** (gpt_neox), trained context 2048, 1.41 B params

| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max abs logit diff |
|---|---|---|---|---|---|---|---|
| offset | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.602 -> 0.602 | 0.0000 | 0.0000 | 0.0004 |
| offset | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.602 -> 0.602 | 0.0000 | 0.0000 | 0.0004 |
| offset | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.602 -> 0.602 | 0.0000 | 0.0000 | 0.0004 |
| offset | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.600 -> 0.600 | 0.0000 | 0.0000 | 0.0004 |
| leftpad | p16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.602 -> 0.602 | 0.0000 | 0.0000 | 0.0004 |
| leftpad | p64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.602 -> 0.602 | 0.0000 | 0.0000 | 0.0004 |
| leftpad | p256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.602 -> 0.602 | 0.0000 | 0.0000 | 0.0004 |
| leftpad | p1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.600 -> 0.600 | 0.0000 | 0.0000 | 0.0004 |
| cache | block16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.602 -> 0.602 | 0.0000 | 0.0000 | 0.0004 |
| cache | block64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.602 -> 0.602 | 0.0000 | 0.0000 | 0.0004 |
| cache | block256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.602 -> 0.602 | 0.0000 | 0.0000 | 0.0004 |
| cache | block1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.600 -> 0.600 | 0.0000 | 0.0000 | 0.0004 |
| cache | gap16 | degrades | -0.1153 [-0.1297, -0.1017] | 0.602 -> 0.486 | 0.4497 | 0.3110 | 3 |
| cache | gap64 | degrades | -0.1467 [-0.1617, -0.1313] | 0.602 -> 0.455 | 0.5307 | 0.3523 | 3.72 |
| cache | gap256 | degrades | -0.1937 [-0.2093, -0.1780] | 0.602 -> 0.408 | 0.5940 | 0.3613 | 4.27 |
| cache | gap1024 | degrades | -0.3136 [-0.3312, -0.2964] | 0.600 -> 0.286 | 0.6948 | 0.4312 | 5.58 |
| window | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.589 -> 0.589 | 0.0000 | 0.0000 | 0.0004 |
| window | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.589 -> 0.589 | 0.0000 | 0.0000 | 0.0004 |
| window | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.589 -> 0.589 | 0.0000 | 0.0000 | 0.0004 |
| window | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.589 -> 0.589 | 0.0000 | 0.0000 | 0.0004 |
| offset_bf16 | k16 | unchanged | -0.0017 [-0.0113, +0.0083] | 0.583 -> 0.581 | 0.2080 | 0.1323 | 2.44 |
| offset_bf16 | k64 | unchanged | -0.0020 [-0.0123, +0.0083] | 0.583 -> 0.581 | 0.2213 | 0.1400 | 2.42 |
| offset_bf16 | k256 | unchanged | -0.0017 [-0.0117, +0.0080] | 0.583 -> 0.581 | 0.2023 | 0.1297 | 2.42 |
| offset_bf16 | k1024 | unchanged | +0.0016 [-0.0080, +0.0112] | 0.586 -> 0.587 | 0.1840 | 0.0856 | 2.44 |

**Serving recommendations (descriptive)**

- shift all position ids by a constant: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- left-pad with naive absolute position ids: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- reuse a prefix KV cache computed with all position indices shifted (block): accuracy unchanged at every tested k (max answer-flip rate 0.000)
- append to a cached prefix with a position gap: accuracy changes at gap16, gap64, gap256, gap1024 -> recompute/reset positions
- start a 128-token sliding window at a non-zero position: accuracy unchanged at every tested k (max answer-flip rate 0.000)

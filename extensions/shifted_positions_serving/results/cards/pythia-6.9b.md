# Answers under shifted position indices: pythia-6.9b

- Hugging Face Hub model `EleutherAI/pythia-6.9b`; license: Apache-2.0
- position class **RoPE** (gpt_neox), trained context 2048, 6.86 B params

| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max abs logit diff |
|---|---|---|---|---|---|---|---|
| offset | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.661 -> 0.661 | 0.0000 | 0.0000 | 0.0007 |
| offset | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.661 -> 0.661 | 0.0000 | 0.0000 | 0.0007 |
| offset | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.661 -> 0.661 | 0.0000 | 0.0000 | 0.0007 |
| offset | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.687 -> 0.687 | 0.0000 | 0.0000 | 0.0006 |
| leftpad | p16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.661 -> 0.661 | 0.0003 | 0.0000 | 0.0007 |
| leftpad | p64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.661 -> 0.661 | 0.0003 | 0.0000 | 0.0007 |
| leftpad | p256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.661 -> 0.661 | 0.0003 | 0.0000 | 0.0007 |
| leftpad | p1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.687 -> 0.687 | 0.0004 | 0.0000 | 0.0007 |
| cache | block16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.661 -> 0.661 | 0.0000 | 0.0000 | 0.0005 |
| cache | block64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.661 -> 0.661 | 0.0000 | 0.0000 | 0.0005 |
| cache | block256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.661 -> 0.661 | 0.0000 | 0.0000 | 0.0005 |
| cache | block1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.687 -> 0.687 | 0.0000 | 0.0000 | 0.0005 |
| cache | gap16 | degrades | -0.1407 [-0.1553, -0.1260] | 0.661 -> 0.520 | 0.4577 | 0.2927 | 4.45 |
| cache | gap64 | degrades | -0.1790 [-0.1943, -0.1633] | 0.661 -> 0.482 | 0.5253 | 0.3183 | 4.51 |
| cache | gap256 | degrades | -0.2307 [-0.2467, -0.2147] | 0.661 -> 0.430 | 0.5903 | 0.3493 | 4.8 |
| cache | gap1024 | degrades | -0.3700 [-0.3868, -0.3532] | 0.687 -> 0.317 | 0.6624 | 0.4092 | 6.55 |
| window | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.666 -> 0.666 | 0.0000 | 0.0000 | 0.0006 |
| window | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.666 -> 0.666 | 0.0000 | 0.0000 | 0.0006 |
| window | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.666 -> 0.666 | 0.0000 | 0.0000 | 0.0006 |
| window | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.666 -> 0.666 | 0.0000 | 0.0000 | 0.0006 |
| offset_bf16 | k16 | unchanged | -0.0030 [-0.0133, +0.0073] | 0.630 -> 0.627 | 0.2137 | 0.1340 | 2.58 |
| offset_bf16 | k64 | unchanged | +0.0087 [-0.0017, +0.0193] | 0.630 -> 0.639 | 0.2067 | 0.1300 | 2.61 |
| offset_bf16 | k256 | unchanged | +0.0000 [-0.0103, +0.0103] | 0.630 -> 0.630 | 0.2070 | 0.1247 | 2.62 |
| offset_bf16 | k1024 | unchanged | -0.0084 [-0.0184, +0.0016] | 0.676 -> 0.667 | 0.1552 | 0.0596 | 2.75 |

**Serving recommendations (descriptive)**

- shift all position ids by a constant: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- left-pad with naive absolute position ids: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- reuse a prefix KV cache computed with all position indices shifted (block): accuracy unchanged at every tested k (max answer-flip rate 0.000)
- append to a cached prefix with a position gap: accuracy changes at gap16, gap64, gap256, gap1024 -> recompute/reset positions
- start a 128-token sliding window at a non-zero position: accuracy unchanged at every tested k (max answer-flip rate 0.000)

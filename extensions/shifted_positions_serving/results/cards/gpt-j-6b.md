# Answers under shifted position indices: gpt-j-6b

- Hugging Face Hub model `EleutherAI/gpt-j-6b`; license: Apache-2.0
- position class **RoPE** (gptj), trained context 2048, 6.05 B params

| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max abs logit diff |
|---|---|---|---|---|---|---|---|
| offset | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.710 -> 0.710 | 0.0000 | 0.0000 | 0 |
| offset | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.710 -> 0.710 | 0.0000 | 0.0000 | 0 |
| offset | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.710 -> 0.710 | 0.0000 | 0.0000 | 0 |
| offset | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.700 -> 0.700 | 0.0000 | 0.0000 | 0 |
| leftpad | p16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.710 -> 0.710 | 0.0000 | 0.0000 | 0 |
| leftpad | p64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.710 -> 0.710 | 0.0000 | 0.0000 | 0 |
| leftpad | p256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.710 -> 0.710 | 0.0000 | 0.0000 | 0 |
| leftpad | p1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.700 -> 0.700 | 0.0000 | 0.0000 | 0 |
| cache | block16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.710 -> 0.710 | 0.0000 | 0.0000 | 0 |
| cache | block64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.710 -> 0.710 | 0.0000 | 0.0000 | 0 |
| cache | block256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.710 -> 0.710 | 0.0000 | 0.0000 | 0 |
| cache | block1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.700 -> 0.700 | 0.0000 | 0.0000 | 0 |
| cache | gap16 | degrades | -0.1657 [-0.1803, -0.1510] | 0.710 -> 0.545 | 0.3780 | 0.1983 | 4.76 |
| cache | gap64 | degrades | -0.2597 [-0.2750, -0.2440] | 0.710 -> 0.451 | 0.4977 | 0.2427 | 4.23 |
| cache | gap256 | degrades | -0.2800 [-0.2953, -0.2647] | 0.710 -> 0.430 | 0.5393 | 0.2833 | 4.95 |
| cache | gap1024 | degrades | -0.2880 [-0.3036, -0.2720] | 0.700 -> 0.412 | 0.5636 | 0.3252 | 5.85 |
| window | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.675 -> 0.675 | 0.0000 | 0.0000 | 0 |
| window | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.675 -> 0.675 | 0.0000 | 0.0000 | 0 |
| window | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.675 -> 0.675 | 0.0000 | 0.0000 | 0 |
| window | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.675 -> 0.675 | 0.0000 | 0.0000 | 0 |
| offset_bf16 | k16 | unchanged | -0.0020 [-0.0043, +0.0003] | 0.708 -> 0.706 | 0.0160 | 0.0063 | 0.125 |
| offset_bf16 | k64 | unchanged | -0.0017 [-0.0040, +0.0007] | 0.708 -> 0.707 | 0.0153 | 0.0067 | 0.125 |
| offset_bf16 | k256 | unchanged | +0.0003 [-0.0020, +0.0027] | 0.708 -> 0.709 | 0.0150 | 0.0057 | 0.125 |
| offset_bf16 | k1024 | unchanged | -0.0020 [-0.0052, +0.0008] | 0.700 -> 0.698 | 0.0184 | 0.0052 | 0.125 |

**Serving recommendations (descriptive)**

- shift all position ids by a constant: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- left-pad with naive absolute position ids: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- reuse a prefix KV cache computed with all position indices shifted (block): accuracy unchanged at every tested k (max answer-flip rate 0.000)
- append to a cached prefix with a position gap: accuracy changes at gap16, gap64, gap256, gap1024 -> recompute/reset positions
- start a 128-token sliding window at a non-zero position: accuracy unchanged at every tested k (max answer-flip rate 0.000)

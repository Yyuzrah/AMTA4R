# Answers under shifted position indices: gpt2

- Hugging Face Hub model `gpt2`; license: MIT
- position class **APE** (gpt2), trained context 1024, 0.124 B params

| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max abs logit diff |
|---|---|---|---|---|---|---|---|
| offset | k16 | degrades | -0.2307 [-0.2437, -0.2187] | 0.276 -> 0.045 | 0.9487 | 0.7307 | 102 |
| offset | k64 | degrades | -0.2433 [-0.2557, -0.2310] | 0.276 -> 0.033 | 0.9627 | 0.7400 | 65.1 |
| offset | k256 | degrades | -0.2723 [-0.2840, -0.2607] | 0.276 -> 0.004 | 0.9817 | 0.7740 | 98 |
| leftpad | p16 | degrades | -0.2307 [-0.2433, -0.2183] | 0.276 -> 0.045 | 0.9487 | 0.7307 | 102 |
| leftpad | p64 | degrades | -0.2433 [-0.2557, -0.2310] | 0.276 -> 0.033 | 0.9627 | 0.7400 | 65.1 |
| leftpad | p256 | degrades | -0.2723 [-0.2843, -0.2610] | 0.276 -> 0.004 | 0.9817 | 0.7740 | 98 |
| cache | block16 | degrades | -0.2307 [-0.2427, -0.2187] | 0.276 -> 0.045 | 0.9487 | 0.7307 | 102 |
| cache | block64 | degrades | -0.2433 [-0.2557, -0.2307] | 0.276 -> 0.033 | 0.9627 | 0.7400 | 65.1 |
| cache | block256 | degrades | -0.2723 [-0.2840, -0.2607] | 0.276 -> 0.004 | 0.9817 | 0.7740 | 98 |
| cache | gap16 | unchanged | -0.0013 [-0.0080, +0.0053] | 0.276 -> 0.275 | 0.2023 | 0.1000 | 4.28 |
| cache | gap64 | degrades | -0.2567 [-0.2680, -0.2453] | 0.276 -> 0.019 | 0.8930 | 0.7277 | 53.4 |
| cache | gap256 | degrades | -0.2663 [-0.2780, -0.2547] | 0.276 -> 0.010 | 0.9350 | 0.8470 | 66.1 |
| window | k16 | degrades | -0.1924 [-0.2064, -0.1788] | 0.196 -> 0.004 | 0.9788 | 0.7716 | 153 |
| window | k64 | degrades | -0.1920 [-0.2060, -0.1780] | 0.196 -> 0.004 | 0.9868 | 0.7816 | 133 |
| window | k256 | degrades | -0.1940 [-0.2076, -0.1804] | 0.196 -> 0.002 | 0.9816 | 0.7928 | 122 |
| offset_bf16 | k16 | degrades | -0.2327 [-0.2450, -0.2197] | 0.276 -> 0.043 | 0.9540 | 0.7493 | 103 |
| offset_bf16 | k64 | degrades | -0.2403 [-0.2527, -0.2280] | 0.276 -> 0.036 | 0.9637 | 0.7657 | 66.8 |
| offset_bf16 | k256 | degrades | -0.2710 [-0.2830, -0.2593] | 0.276 -> 0.005 | 0.9860 | 0.7780 | 96.9 |

**Serving recommendations (descriptive)**

- shift all position ids by a constant: accuracy changes at k16, k64, k256 -> recompute/reset positions
- left-pad with naive absolute position ids: accuracy changes at p16, p64, p256 -> recompute/reset positions
- reuse a prefix KV cache computed with all position indices shifted (block): accuracy changes at block16, block64, block256 -> recompute/reset positions
- append to a cached prefix with a position gap: accuracy changes at gap64, gap256 -> recompute/reset positions
- start a 128-token sliding window at a non-zero position: accuracy changes at k16, k64, k256 -> recompute/reset positions

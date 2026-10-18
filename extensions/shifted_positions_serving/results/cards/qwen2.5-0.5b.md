# Answers under shifted position indices: qwen2.5-0.5b

- Hugging Face Hub model `Qwen/Qwen2.5-0.5B`; license: Apache-2.0
- position class **RoPE** (qwen2), trained context 32768, 0.494 B params

| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max abs logit diff |
|---|---|---|---|---|---|---|---|
| offset | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.635 -> 0.635 | 0.0003 | 0.0003 | 0 |
| offset | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.635 -> 0.635 | 0.0000 | 0.0000 | 0 |
| offset | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.635 -> 0.635 | 0.0000 | 0.0000 | 0 |
| offset | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.635 -> 0.635 | 0.0000 | 0.0000 | 0.0001 |
| leftpad | p16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.635 -> 0.635 | 0.0003 | 0.0003 | 0 |
| leftpad | p64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.635 -> 0.635 | 0.0000 | 0.0000 | 0 |
| leftpad | p256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.635 -> 0.635 | 0.0000 | 0.0000 | 0 |
| leftpad | p1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.635 -> 0.635 | 0.0000 | 0.0000 | 0.0001 |
| cache | block16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.635 -> 0.635 | 0.0003 | 0.0003 | 0 |
| cache | block64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.635 -> 0.635 | 0.0000 | 0.0000 | 0 |
| cache | block256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.635 -> 0.635 | 0.0000 | 0.0000 | 0 |
| cache | block1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.635 -> 0.635 | 0.0000 | 0.0000 | 0.0001 |
| cache | gap16 | degrades | -0.1300 [-0.1430, -0.1167] | 0.635 -> 0.505 | 0.4233 | 0.1337 | 5.09 |
| cache | gap64 | degrades | -0.1613 [-0.1747, -0.1480] | 0.635 -> 0.474 | 0.4780 | 0.1530 | 6.01 |
| cache | gap256 | degrades | -0.1760 [-0.1907, -0.1617] | 0.635 -> 0.459 | 0.5257 | 0.1657 | 6.27 |
| cache | gap1024 | degrades | -0.2007 [-0.2160, -0.1857] | 0.635 -> 0.434 | 0.5627 | 0.2000 | 6.54 |
| window | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.610 -> 0.610 | 0.0000 | 0.0000 | 0 |
| window | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.610 -> 0.610 | 0.0000 | 0.0000 | 0 |
| window | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.610 -> 0.610 | 0.0000 | 0.0000 | 0 |
| window | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.610 -> 0.610 | 0.0000 | 0.0000 | 0 |
| offset_bf16 | k16 | unchanged | +0.0003 [-0.0050, +0.0053] | 0.631 -> 0.632 | 0.0570 | 0.0260 | 0.34 |
| offset_bf16 | k64 | unchanged | -0.0007 [-0.0060, +0.0050] | 0.631 -> 0.631 | 0.0553 | 0.0257 | 0.344 |
| offset_bf16 | k256 | unchanged | +0.0017 [-0.0033, +0.0067] | 0.631 -> 0.633 | 0.0627 | 0.0310 | 0.383 |
| offset_bf16 | k1024 | unchanged | -0.0010 [-0.0063, +0.0043] | 0.631 -> 0.630 | 0.0607 | 0.0280 | 0.406 |

**Serving recommendations (descriptive)**

- shift all position ids by a constant: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- left-pad with naive absolute position ids: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- reuse a prefix KV cache computed with all position indices shifted (block): accuracy unchanged at every tested k (max answer-flip rate 0.000)
- append to a cached prefix with a position gap: accuracy changes at gap16, gap64, gap256, gap1024 -> recompute/reset positions
- start a 128-token sliding window at a non-zero position: accuracy unchanged at every tested k (max answer-flip rate 0.000)

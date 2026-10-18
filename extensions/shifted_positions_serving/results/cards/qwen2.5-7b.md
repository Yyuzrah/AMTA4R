# Answers under shifted position indices: qwen2.5-7b

- Hugging Face Hub model `Qwen/Qwen2.5-7B`; license: Apache-2.0
- position class **RoPE** (qwen2), trained context 32768, 7.62 B params

| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max abs logit diff |
|---|---|---|---|---|---|---|---|
| offset | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.818 -> 0.818 | 0.0000 | 0.0000 | 0 |
| offset | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.818 -> 0.818 | 0.0000 | 0.0000 | 0 |
| offset | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.818 -> 0.818 | 0.0000 | 0.0000 | 0 |
| offset | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.818 -> 0.818 | 0.0000 | 0.0000 | 0.0001 |
| leftpad | p16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.818 -> 0.818 | 0.0000 | 0.0000 | 0 |
| leftpad | p64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.818 -> 0.818 | 0.0000 | 0.0000 | 0.0001 |
| leftpad | p256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.818 -> 0.818 | 0.0000 | 0.0000 | 0 |
| leftpad | p1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.818 -> 0.818 | 0.0000 | 0.0000 | 0.0001 |
| cache | block16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.818 -> 0.818 | 0.0000 | 0.0000 | 0 |
| cache | block64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.818 -> 0.818 | 0.0000 | 0.0000 | 0 |
| cache | block256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.818 -> 0.818 | 0.0000 | 0.0000 | 0 |
| cache | block1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.818 -> 0.818 | 0.0000 | 0.0000 | 0.0001 |
| cache | gap16 | degrades | -0.1403 [-0.1507, -0.1303] | 0.818 -> 0.677 | 0.3173 | 0.0777 | 5.93 |
| cache | gap64 | degrades | -0.1527 [-0.1633, -0.1423] | 0.818 -> 0.665 | 0.3427 | 0.0840 | 5.81 |
| cache | gap256 | degrades | -0.1633 [-0.1750, -0.1517] | 0.818 -> 0.654 | 0.3620 | 0.0897 | 5.62 |
| cache | gap1024 | degrades | -0.1763 [-0.1890, -0.1637] | 0.818 -> 0.641 | 0.3747 | 0.0947 | 5.56 |
| window | k16 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.782 -> 0.782 | 0.0000 | 0.0000 | 0 |
| window | k64 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.782 -> 0.782 | 0.0000 | 0.0000 | 0 |
| window | k256 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.782 -> 0.782 | 0.0000 | 0.0000 | 0 |
| window | k1024 | unchanged | +0.0000 [+0.0000, +0.0000] | 0.782 -> 0.782 | 0.0000 | 0.0000 | 0.0001 |
| offset_bf16 | k16 | unchanged | -0.0037 [-0.0097, +0.0023] | 0.684 -> 0.680 | 0.0880 | 0.0547 | 0.715 |
| offset_bf16 | k64 | degrades | -0.0133 [-0.0213, -0.0057] | 0.684 -> 0.671 | 0.1200 | 0.0833 | 0.75 |
| offset_bf16 | k256 | degrades | -0.0223 [-0.0313, -0.0133] | 0.684 -> 0.662 | 0.1530 | 0.0963 | 0.971 |
| offset_bf16 | k1024 | unchanged | +0.0133 [+0.0000, +0.0267] | 0.684 -> 0.697 | 0.2817 | 0.1383 | 2.39 |

**Serving recommendations (descriptive)**

- shift all position ids by a constant: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- left-pad with naive absolute position ids: accuracy unchanged at every tested k (max answer-flip rate 0.000)
- reuse a prefix KV cache computed with all position indices shifted (block): accuracy unchanged at every tested k (max answer-flip rate 0.000)
- append to a cached prefix with a position gap: accuracy changes at gap16, gap64, gap256, gap1024 -> recompute/reset positions
- start a 128-token sliding window at a non-zero position: accuracy unchanged at every tested k (max answer-flip rate 0.000)

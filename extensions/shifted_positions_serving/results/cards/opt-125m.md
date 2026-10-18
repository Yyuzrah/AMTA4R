# Answers under shifted position indices: opt-125m

- Hugging Face Hub model `facebook/opt-125m`; license: OPT-175B License Agreement (hub tag "other"; non-commercial research use permitted)
- position class **APE** (opt), trained context 2048, 0.125 B params

| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max abs logit diff |
|---|---|---|---|---|---|---|---|
| offset | k16 | degrades | -0.1310 [-0.1407, -0.1210] | 0.131 -> 0.000 | 0.9993 | 0.9837 | 11.1 |
| offset | k64 | degrades | -0.1207 [-0.1313, -0.1103] | 0.131 -> 0.010 | 0.9183 | 0.7397 | 6.83 |
| offset | k256 | degrades | -0.1310 [-0.1407, -0.1213] | 0.131 -> 0.000 | 0.9967 | 0.8397 | 12.6 |
| offset | k1024 | degrades | -0.1372 [-0.1476, -0.1264] | 0.138 -> 0.000 | 0.9996 | 0.9900 | 10.8 |
| leftpad | p16 | degrades | -0.1310 [-0.1410, -0.1213] | 0.131 -> 0.000 | 0.9993 | 0.9837 | 11.1 |
| leftpad | p64 | degrades | -0.1207 [-0.1313, -0.1100] | 0.131 -> 0.010 | 0.9183 | 0.7397 | 6.83 |
| leftpad | p256 | degrades | -0.1310 [-0.1410, -0.1210] | 0.131 -> 0.000 | 0.9967 | 0.8397 | 12.6 |
| leftpad | p1024 | degrades | -0.1372 [-0.1480, -0.1264] | 0.138 -> 0.000 | 0.9996 | 0.9900 | 10.8 |
| cache | block16 | degrades | -0.1310 [-0.1410, -0.1213] | 0.131 -> 0.000 | 0.9993 | 0.9837 | 11.1 |
| cache | block64 | degrades | -0.1207 [-0.1310, -0.1103] | 0.131 -> 0.010 | 0.9183 | 0.7397 | 6.83 |
| cache | block256 | degrades | -0.1310 [-0.1410, -0.1210] | 0.131 -> 0.000 | 0.9967 | 0.8397 | 12.6 |
| cache | block1024 | degrades | -0.1372 [-0.1484, -0.1268] | 0.138 -> 0.000 | 0.9996 | 0.9900 | 10.8 |
| cache | gap16 | unchanged | +0.0067 [-0.0010, +0.0140] | 0.131 -> 0.138 | 0.4117 | 0.2477 | 1.41 |
| cache | gap64 | degrades | -0.0327 [-0.0417, -0.0237] | 0.131 -> 0.098 | 0.5510 | 0.3443 | 2.45 |
| cache | gap256 | degrades | -0.1000 [-0.1093, -0.0903] | 0.131 -> 0.031 | 0.7183 | 0.5690 | 3.63 |
| cache | gap1024 | degrades | -0.1244 [-0.1352, -0.1136] | 0.138 -> 0.013 | 0.9464 | 0.7688 | 7.71 |
| window | k16 | degrades | -0.1440 [-0.1544, -0.1336] | 0.144 -> 0.000 | 0.9916 | 0.9748 | 11.1 |
| window | k64 | degrades | -0.1348 [-0.1456, -0.1240] | 0.144 -> 0.009 | 0.8888 | 0.6904 | 6.76 |
| window | k256 | degrades | -0.1440 [-0.1548, -0.1336] | 0.144 -> 0.000 | 0.9900 | 0.8120 | 13 |
| window | k1024 | degrades | -0.1436 [-0.1540, -0.1332] | 0.144 -> 0.000 | 0.9984 | 0.9880 | 10.5 |
| offset_bf16 | k16 | degrades | -0.1337 [-0.1437, -0.1237] | 0.134 -> 0.000 | 0.9993 | 0.9873 | 11.1 |
| offset_bf16 | k64 | degrades | -0.1233 [-0.1337, -0.1130] | 0.134 -> 0.010 | 0.9140 | 0.7247 | 6.64 |
| offset_bf16 | k256 | degrades | -0.1337 [-0.1433, -0.1240] | 0.134 -> 0.000 | 0.9967 | 0.8390 | 12.7 |
| offset_bf16 | k1024 | degrades | -0.1396 [-0.1504, -0.1292] | 0.140 -> 0.000 | 0.9996 | 0.9912 | 10.8 |

**Serving recommendations (descriptive)**

- shift all position ids by a constant: accuracy changes at k16, k64, k256, k1024 -> recompute/reset positions
- left-pad with naive absolute position ids: accuracy changes at p16, p64, p256, p1024 -> recompute/reset positions
- reuse a prefix KV cache computed with all position indices shifted (block): accuracy changes at block16, block64, block256, block1024 -> recompute/reset positions
- append to a cached prefix with a position gap: accuracy changes at gap64, gap256, gap1024 -> recompute/reset positions
- start a 128-token sliding window at a non-zero position: accuracy changes at k16, k64, k256, k1024 -> recompute/reset positions

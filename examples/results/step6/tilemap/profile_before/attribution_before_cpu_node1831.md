# Craftax tile/atlas phase attribution — before

- tree measured: `/oscar/scratch/sboughan/se6/baseline/stable_envs/__init__.py`
- pipeline present in that tree: **blend**
- host: node1831.oscar.ccv.brown.edu  ·  cpu: x86_64  ·  gpu: none  ·  device: cpu  ·  threads: 1
- slurm job: 6349384  ·  load average: 39.4
- caches: L1d cache:                               1.5 MiB (48 instances); L1i cache:                               1.5 MiB (48 instances); L2 cache:                                48 MiB (48 instances); L3 cache:                                71.5 MiB (2 instances)
- torch 2.14.0+cu126  ·  tile size 16 px

## batch 1

`env.render` **0.798 ms** = **1253 fps**  ·  set_state 0.041 ms

| phase | ms median | % of render | frames/s if this were all |
|---|---|---|---|
| decode (atlas indexing) | 0.053 | 6.6 | 18863 |
| blend: tile gather | 0.049 | 6.1 | 20500 |
| blend: gather + layout | 0.068 | 8.5 | 14660 |
| blend: player composite | 0.051 | 6.4 | 19475 |
| blend: mob alpha layer | 0.091 | 11.4 | 10966 |
| blend: mob composite | 0.166 | 20.8 | 6013 |
| blend: channel gain (recolour) | 0.063 | 7.9 | 15864 |
| blend: output conversion | 0.045 | 5.6 | 22174 |
| blend: whole pipeline | 0.529 | 66.3 | 1889 |
| allocation: empty uint8 frame | 0.003 | 0.3 | 378778 |
| allocation: empty float frame | 0.003 | 0.3 | 368730 |

image-sized float32 temporaries in the blend pipeline: 1.94 MB per frame, 1.9 MB per batch, against a 48.4 KB frame

## batch 16

`env.render` **8.527 ms** = **1876 fps**  ·  set_state 0.046 ms

| phase | ms median | % of render | frames/s if this were all |
|---|---|---|---|
| decode (atlas indexing) | 0.062 | 0.7 | 259248 |
| blend: tile gather | 0.719 | 8.4 | 22268 |
| blend: gather + layout | 1.018 | 11.9 | 15717 |
| blend: player composite | 0.089 | 1.0 | 179436 |
| blend: mob alpha layer | 1.507 | 17.7 | 10620 |
| blend: mob composite | 3.192 | 37.4 | 5013 |
| blend: channel gain (recolour) | 0.828 | 9.7 | 19328 |
| blend: output conversion | 0.787 | 9.2 | 20318 |
| blend: whole pipeline | 8.230 | 96.5 | 1944 |
| allocation: empty uint8 frame | 0.003 | 0.0 | 5818261 |
| allocation: empty float frame | 0.002 | 0.0 | 7023659 |

image-sized float32 temporaries in the blend pipeline: 1.94 MB per frame, 31.0 MB per batch, against a 48.4 KB frame

## batch 64

`env.render` **38.476 ms** = **1663 fps**  ·  set_state 0.063 ms

| phase | ms median | % of render | frames/s if this were all |
|---|---|---|---|
| decode (atlas indexing) | 0.084 | 0.2 | 758319 |
| blend: tile gather | 2.857 | 7.4 | 22401 |
| blend: gather + layout | 5.137 | 13.4 | 12459 |
| blend: player composite | 0.249 | 0.6 | 257270 |
| blend: mob alpha layer | 7.044 | 18.3 | 9085 |
| blend: mob composite | 15.739 | 40.9 | 4066 |
| blend: channel gain (recolour) | 3.726 | 9.7 | 17178 |
| blend: output conversion | 3.395 | 8.8 | 18854 |
| blend: whole pipeline | 40.972 | 106.5 | 1562 |
| allocation: empty uint8 frame | 0.002 | 0.0 | 25701534 |
| allocation: empty float frame | 0.002 | 0.0 | 27199476 |

image-sized float32 temporaries in the blend pipeline: 1.94 MB per frame, 123.9 MB per batch, against a 48.4 KB frame

## batch 256

`env.render` **230.060 ms** = **1113 fps**  ·  set_state 0.134 ms

| phase | ms median | % of render | frames/s if this were all |
|---|---|---|---|
| decode (atlas indexing) | 0.174 | 0.1 | 1472976 |
| blend: tile gather | 18.725 | 8.1 | 13671 |
| blend: gather + layout | 34.073 | 14.8 | 7513 |
| blend: player composite | 0.971 | 0.4 | 263753 |
| blend: mob alpha layer | 47.882 | 20.8 | 5346 |
| blend: mob composite | 103.786 | 45.1 | 2467 |
| blend: channel gain (recolour) | 22.039 | 9.6 | 11616 |
| blend: output conversion | 25.474 | 11.1 | 10049 |
| blend: whole pipeline | 232.897 | 101.2 | 1099 |
| allocation: empty uint8 frame | 0.002 | 0.0 | 109502204 |
| allocation: empty float frame | 0.009 | 0.0 | 28281075 |

image-sized float32 temporaries in the blend pipeline: 1.94 MB per frame, 495.5 MB per batch, against a 48.4 KB frame

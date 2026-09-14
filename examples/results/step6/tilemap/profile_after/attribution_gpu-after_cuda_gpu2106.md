# Craftax tile/atlas phase attribution — gpu-after

- tree measured: `/oscar/scratch/sboughan/se6/tilemap/stable_envs/__init__.py`
- pipeline present in that tree: **atlas**
- host: gpu2106.oscar.ccv.brown.edu  ·  cpu: x86_64  ·  gpu: NVIDIA GeForce RTX 3090  ·  device: cuda  ·  threads: 4
- slurm job: 6349394  ·  load average: 29.5
- caches: L1d cache:                               2 MiB (64 instances); L1i cache:                               2 MiB (64 instances); L2 cache:                                32 MiB (64 instances); L3 cache:                                512 MiB (32 instances)
- torch 2.14.0+cu126  ·  tile size 16 px

## batch 1

`env.render` **0.592 ms** = **1690 fps**  ·  set_state 0.102 ms

| phase | ms median | % of render | frames/s if this were all |
|---|---|---|---|
| decode (atlas indexing) | 0.131 | 22.1 | 7656 |
| blend: tile gather | 0.023 | 3.9 | 42919 |
| blend: gather + layout | 0.047 | 7.9 | 21306 |
| blend: player composite | 0.126 | 21.3 | 7919 |
| blend: mob alpha layer | 0.059 | 10.0 | 16867 |
| blend: mob composite | 0.098 | 16.6 | 10184 |
| blend: channel gain (recolour) | 0.018 | 3.0 | 55977 |
| blend: output conversion | 0.040 | 6.7 | 25070 |
| blend: whole pipeline | 0.352 | 59.6 | 2838 |
| allocation: empty uint8 frame | 0.011 | 1.9 | 90261 |
| allocation: empty float frame | 0.011 | 1.9 | 90420 |
| atlas: index build | 0.256 | 43.3 | 3907 |
| atlas: gather into uint8 output | 0.024 | 4.1 | 40834 |
| atlas: bake (once per device+size) | 0.213 | 36.0 | 4698 |
| atlas: gain path (whole render) | 0.715 | 120.9 | 1398 |

image-sized float32 temporaries in the blend pipeline: 1.94 MB per frame, 1.9 MB per batch, against a 48.4 KB frame; composited atlas 522 KB

## batch 64

`env.render` **0.634 ms** = **101019 fps**  ·  set_state 0.127 ms

| phase | ms median | % of render | frames/s if this were all |
|---|---|---|---|
| decode (atlas indexing) | 0.136 | 21.5 | 470920 |
| blend: tile gather | 0.032 | 5.1 | 1988756 |
| blend: gather + layout | 0.068 | 10.7 | 940281 |
| blend: player composite | 0.128 | 20.2 | 500142 |
| blend: mob alpha layer | 0.103 | 16.2 | 624121 |
| blend: mob composite | 0.237 | 37.4 | 270022 |
| blend: channel gain (recolour) | 0.044 | 6.9 | 1464141 |
| blend: output conversion | 0.096 | 15.2 | 664669 |
| blend: whole pipeline | 0.605 | 95.4 | 105857 |
| allocation: empty uint8 frame | 0.011 | 1.7 | 5834562 |
| allocation: empty float frame | 0.011 | 1.7 | 5964197 |
| atlas: index build | 0.259 | 40.9 | 246753 |
| atlas: gather into uint8 output | 0.063 | 10.0 | 1009615 |
| atlas: bake (once per device+size) | 0.209 | 33.0 | 306119 |
| atlas: gain path (whole render) | 0.823 | 129.9 | 77762 |

image-sized float32 temporaries in the blend pipeline: 1.94 MB per frame, 123.9 MB per batch, against a 48.4 KB frame; composited atlas 522 KB

## batch 256

`env.render` **0.850 ms** = **301097 fps**  ·  set_state 0.201 ms

| phase | ms median | % of render | frames/s if this were all |
|---|---|---|---|
| decode (atlas indexing) | 0.139 | 16.3 | 1848130 |
| blend: tile gather | 0.076 | 8.9 | 3373066 |
| blend: gather + layout | 0.198 | 23.3 | 1292023 |
| blend: player composite | 0.129 | 15.1 | 1990297 |
| blend: mob alpha layer | 0.319 | 37.5 | 802174 |
| blend: mob composite | 0.850 | 99.9 | 301302 |
| blend: channel gain (recolour) | 0.131 | 15.4 | 1952216 |
| blend: output conversion | 0.329 | 38.7 | 778180 |
| blend: whole pipeline | 1.794 | 211.0 | 142715 |
| allocation: empty uint8 frame | 0.011 | 1.3 | 23958677 |
| allocation: empty float frame | 0.011 | 1.3 | 23550198 |
| atlas: index build | 0.262 | 30.8 | 977483 |
| atlas: gather into uint8 output | 0.184 | 21.7 | 1387768 |
| atlas: bake (once per device+size) | 0.222 | 26.1 | 1155610 |
| atlas: gain path (whole render) | 1.330 | 156.4 | 192503 |

image-sized float32 temporaries in the blend pipeline: 1.94 MB per frame, 495.5 MB per batch, against a 48.4 KB frame; composited atlas 522 KB

## batch 4096

`env.render` **3.831 ms** = **1069105 fps**  ·  set_state 0.676 ms

| phase | ms median | % of render | frames/s if this were all |
|---|---|---|---|
| decode (atlas indexing) | 0.136 | 3.6 | 30054645 |
| blend: tile gather | 0.946 | 24.7 | 4331327 |
| blend: gather + layout | 2.807 | 73.3 | 1459138 |
| blend: player composite | 0.306 | 8.0 | 13401038 |
| blend: mob alpha layer | 4.665 | 121.8 | 878019 |
| blend: mob composite | 13.153 | 343.3 | 311405 |
| blend: channel gain (recolour) | 1.876 | 49.0 | 2182925 |
| blend: output conversion | 4.959 | 129.4 | 825899 |
| blend: whole pipeline | 27.672 | 722.3 | 148018 |
| allocation: empty uint8 frame | 0.011 | 0.3 | 366503876 |
| allocation: empty float frame | 0.011 | 0.3 | 382571896 |
| atlas: index build | 0.262 | 6.8 | 15645954 |
| atlas: gather into uint8 output | 2.595 | 67.7 | 1578617 |
| atlas: bake (once per device+size) | 0.212 | 5.5 | 19344657 |
| atlas: gain path (whole render) | 10.662 | 278.3 | 384162 |

image-sized float32 temporaries in the blend pipeline: 1.94 MB per frame, 7927.2 MB per batch, against a 48.4 KB frame; composited atlas 522 KB

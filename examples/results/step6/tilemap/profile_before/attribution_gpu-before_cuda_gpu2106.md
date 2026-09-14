# Craftax tile/atlas phase attribution — gpu-before

- tree measured: `/oscar/scratch/sboughan/se6/baseline/stable_envs/__init__.py`
- pipeline present in that tree: **blend**
- host: gpu2106.oscar.ccv.brown.edu  ·  cpu: x86_64  ·  gpu: NVIDIA GeForce RTX 3090  ·  device: cuda  ·  threads: 4
- slurm job: 6349394  ·  load average: 28.8
- caches: L1d cache:                               2 MiB (64 instances); L1i cache:                               2 MiB (64 instances); L2 cache:                                32 MiB (64 instances); L3 cache:                                512 MiB (32 instances)
- torch 2.14.0+cu126  ·  tile size 16 px

## batch 1

`env.render` **0.730 ms** = **1370 fps**  ·  set_state 0.102 ms

| phase | ms median | % of render | frames/s if this were all |
|---|---|---|---|
| decode (atlas indexing) | 0.139 | 19.0 | 7219 |
| blend: tile gather | 0.024 | 3.3 | 41102 |
| blend: gather + layout | 0.048 | 6.5 | 21021 |
| blend: player composite | 0.130 | 17.8 | 7716 |
| blend: mob alpha layer | 0.061 | 8.3 | 16527 |
| blend: mob composite | 0.100 | 13.7 | 9985 |
| blend: channel gain (recolour) | 0.019 | 2.6 | 52017 |
| blend: output conversion | 0.043 | 5.8 | 23488 |
| blend: whole pipeline | 0.357 | 48.9 | 2805 |
| allocation: empty uint8 frame | 0.011 | 1.5 | 91157 |
| allocation: empty float frame | 0.011 | 1.4 | 95005 |

image-sized float32 temporaries in the blend pipeline: 1.94 MB per frame, 1.9 MB per batch, against a 48.4 KB frame

## batch 64

`env.render` **0.952 ms** = **67223 fps**  ·  set_state 0.126 ms

| phase | ms median | % of render | frames/s if this were all |
|---|---|---|---|
| decode (atlas indexing) | 0.140 | 14.7 | 457331 |
| blend: tile gather | 0.032 | 3.4 | 1981359 |
| blend: gather + layout | 0.069 | 7.2 | 932839 |
| blend: player composite | 0.130 | 13.6 | 493980 |
| blend: mob alpha layer | 0.103 | 10.8 | 622166 |
| blend: mob composite | 0.237 | 24.9 | 269773 |
| blend: channel gain (recolour) | 0.044 | 4.6 | 1448128 |
| blend: output conversion | 0.097 | 10.2 | 659908 |
| blend: whole pipeline | 0.605 | 63.5 | 105847 |
| allocation: empty uint8 frame | 0.011 | 1.1 | 6014307 |
| allocation: empty float frame | 0.010 | 1.1 | 6186485 |

image-sized float32 temporaries in the blend pipeline: 1.94 MB per frame, 123.9 MB per batch, against a 48.4 KB frame

## batch 256

`env.render` **2.145 ms** = **119323 fps**  ·  set_state 0.198 ms

| phase | ms median | % of render | frames/s if this were all |
|---|---|---|---|
| decode (atlas indexing) | 0.138 | 6.5 | 1848938 |
| blend: tile gather | 0.075 | 3.5 | 3393177 |
| blend: gather + layout | 0.198 | 9.2 | 1295989 |
| blend: player composite | 0.129 | 6.0 | 1977610 |
| blend: mob alpha layer | 0.317 | 14.8 | 807019 |
| blend: mob composite | 0.848 | 39.5 | 301852 |
| blend: channel gain (recolour) | 0.131 | 6.1 | 1948839 |
| blend: output conversion | 0.327 | 15.2 | 783776 |
| blend: whole pipeline | 1.790 | 83.4 | 143011 |
| allocation: empty uint8 frame | 0.010 | 0.5 | 24474927 |
| allocation: empty float frame | 0.010 | 0.5 | 24424907 |

image-sized float32 temporaries in the blend pipeline: 1.94 MB per frame, 495.5 MB per batch, against a 48.4 KB frame

## batch 4096

`env.render` **28.459 ms** = **143927 fps**  ·  set_state 0.673 ms

| phase | ms median | % of render | frames/s if this were all |
|---|---|---|---|
| decode (atlas indexing) | 0.137 | 0.5 | 29893671 |
| blend: tile gather | 0.946 | 3.3 | 4330653 |
| blend: gather + layout | 2.805 | 9.9 | 1460023 |
| blend: player composite | 0.306 | 1.1 | 13369792 |
| blend: mob alpha layer | 4.664 | 16.4 | 878220 |
| blend: mob composite | 13.061 | 45.9 | 313609 |
| blend: channel gain (recolour) | 1.876 | 6.6 | 2183937 |
| blend: output conversion | 4.950 | 17.4 | 827521 |
| blend: whole pipeline | 27.577 | 96.9 | 148531 |
| allocation: empty uint8 frame | 0.011 | 0.0 | 379043912 |
| allocation: empty float frame | 0.011 | 0.0 | 388657345 |

image-sized float32 temporaries in the blend pipeline: 1.94 MB per frame, 7927.2 MB per batch, against a 48.4 KB frame

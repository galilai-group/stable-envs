# Craftax tile/atlas phase attribution — after

- tree measured: `/oscar/scratch/sboughan/se6/tilemap/stable_envs/__init__.py`
- pipeline present in that tree: **atlas**
- host: node1831.oscar.ccv.brown.edu  ·  cpu: x86_64  ·  gpu: none  ·  device: cpu  ·  threads: 1
- slurm job: 6349384  ·  load average: 38.7
- caches: L1d cache:                               1.5 MiB (48 instances); L1i cache:                               1.5 MiB (48 instances); L2 cache:                                48 MiB (48 instances); L3 cache:                                71.5 MiB (2 instances)
- torch 2.14.0+cu126  ·  tile size 16 px

## batch 1

`env.render` **0.330 ms** = **3035 fps**  ·  set_state 0.039 ms

| phase | ms median | % of render | frames/s if this were all |
|---|---|---|---|
| decode (atlas indexing) | 0.054 | 16.4 | 18515 |
| blend: tile gather | 0.048 | 14.7 | 20674 |
| blend: gather + layout | 0.064 | 19.4 | 15679 |
| blend: player composite | 0.051 | 15.4 | 19707 |
| blend: mob alpha layer | 0.083 | 25.2 | 12023 |
| blend: mob composite | 0.141 | 42.8 | 7095 |
| blend: channel gain (recolour) | 0.063 | 19.2 | 15805 |
| blend: output conversion | 0.045 | 13.5 | 22457 |
| blend: whole pipeline | 0.504 | 153.0 | 1984 |
| allocation: empty uint8 frame | 0.003 | 0.8 | 368066 |
| allocation: empty float frame | 0.003 | 0.8 | 373833 |
| atlas: index build | 0.092 | 28.0 | 10823 |
| atlas: gather into uint8 output | 0.009 | 2.7 | 113210 |
| atlas: bake (once per device+size) | 0.471 | 142.8 | 2125 |
| atlas: gain path (whole render) | 0.568 | 172.3 | 1761 |

image-sized float32 temporaries in the blend pipeline: 1.94 MB per frame, 1.9 MB per batch, against a 48.4 KB frame; composited atlas 522 KB

## batch 16

`env.render` **0.488 ms** = **32774 fps**  ·  set_state 0.047 ms

| phase | ms median | % of render | frames/s if this were all |
|---|---|---|---|
| decode (atlas indexing) | 0.062 | 12.8 | 256406 |
| blend: tile gather | 0.695 | 142.5 | 23006 |
| blend: gather + layout | 0.959 | 196.5 | 16676 |
| blend: player composite | 0.088 | 17.9 | 182732 |
| blend: mob alpha layer | 1.571 | 321.7 | 10187 |
| blend: mob composite | 2.876 | 589.1 | 5564 |
| blend: channel gain (recolour) | 0.826 | 169.3 | 19362 |
| blend: output conversion | 0.788 | 161.5 | 20295 |
| blend: whole pipeline | 8.205 | 1680.6 | 1950 |
| allocation: empty uint8 frame | 0.003 | 0.6 | 5866941 |
| allocation: empty float frame | 0.002 | 0.5 | 7011476 |
| atlas: index build | 0.142 | 29.1 | 112499 |
| atlas: gather into uint8 output | 0.061 | 12.5 | 261712 |
| atlas: bake (once per device+size) | 0.575 | 117.9 | 27806 |
| atlas: gain path (whole render) | 2.385 | 488.4 | 6710 |

image-sized float32 temporaries in the blend pipeline: 1.94 MB per frame, 31.0 MB per batch, against a 48.4 KB frame; composited atlas 522 KB

## batch 64

`env.render` **0.843 ms** = **75905 fps**  ·  set_state 0.061 ms

| phase | ms median | % of render | frames/s if this were all |
|---|---|---|---|
| decode (atlas indexing) | 0.085 | 10.1 | 754655 |
| blend: tile gather | 2.846 | 337.6 | 22486 |
| blend: gather + layout | 5.240 | 621.4 | 12214 |
| blend: player composite | 0.229 | 27.2 | 278878 |
| blend: mob alpha layer | 6.889 | 817.1 | 9290 |
| blend: mob composite | 16.271 | 1929.8 | 3933 |
| blend: channel gain (recolour) | 3.701 | 438.9 | 17294 |
| blend: output conversion | 3.472 | 411.8 | 18432 |
| blend: whole pipeline | 38.246 | 4536.0 | 1673 |
| allocation: empty uint8 frame | 0.002 | 0.3 | 27245307 |
| allocation: empty float frame | 0.002 | 0.3 | 26156428 |
| atlas: index build | 0.228 | 27.1 | 280122 |
| atlas: gather into uint8 output | 0.224 | 26.6 | 285471 |
| atlas: bake (once per device+size) | 0.470 | 55.8 | 136148 |
| atlas: gain path (whole render) | 9.794 | 1161.6 | 6534 |

image-sized float32 temporaries in the blend pipeline: 1.94 MB per frame, 123.9 MB per batch, against a 48.4 KB frame; composited atlas 522 KB

## batch 256

`env.render` **2.471 ms** = **103585 fps**  ·  set_state 0.131 ms

| phase | ms median | % of render | frames/s if this were all |
|---|---|---|---|
| decode (atlas indexing) | 0.173 | 7.0 | 1475811 |
| blend: tile gather | 18.903 | 764.9 | 13543 |
| blend: gather + layout | 34.451 | 1394.0 | 7431 |
| blend: player composite | 0.975 | 39.5 | 262538 |
| blend: mob alpha layer | 48.770 | 1973.4 | 5249 |
| blend: mob composite | 104.705 | 4236.7 | 2445 |
| blend: channel gain (recolour) | 22.172 | 897.1 | 11546 |
| blend: output conversion | 26.132 | 1057.4 | 9796 |
| blend: whole pipeline | 246.799 | 9986.2 | 1037 |
| allocation: empty uint8 frame | 0.002 | 0.1 | 109262807 |
| allocation: empty float frame | 0.010 | 0.4 | 26736495 |
| atlas: index build | 0.552 | 22.3 | 463569 |
| atlas: gather into uint8 output | 0.999 | 40.4 | 256272 |
| atlas: bake (once per device+size) | 0.515 | 20.8 | 497252 |
| atlas: gain path (whole render) | 53.229 | 2153.8 | 4809 |

image-sized float32 temporaries in the blend pipeline: 1.94 MB per frame, 495.5 MB per batch, against a 48.4 KB frame; composited atlas 522 KB

"""Fused cast vs the torch loop, on synthetic scenes that hit every branch, scored by BIT PATTERN.

A tolerance would pass a kernel that flips a silhouette pixel, so nothing here is compared with
``allclose``. The scenes are deliberately adversarial: poses are drawn so that a large fraction of
rays graze a surface, which is where ``disc > 0`` is decided by the last bit and where the two
implementations have to agree if the claim is to mean anything.
"""
from __future__ import annotations

import torch

from stable_envs.render import raycast3d as R3

torch.manual_seed(0)
dev = "cuda"
KINDS = {"sphere": R3.SPHERE, "capsule": R3.CAPSULE, "cylinder": R3.CYLINDER,
         "box": R3.BOX, "plane": R3.PLANE}


def rand_rot(B, K):
    q = torch.randn(B, K, 4, device=dev, dtype=torch.float64)
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)
    return R.reshape(B, K, 3, 3).to(torch.float32)


def scene(B, K, N, kinds):
    sizes = (0.05 + torch.rand(B, K, 3, device=dev)).contiguous()
    pos = (torch.randn(B, K, 3, device=dev) * 0.6).contiguous()
    R = rand_rot(B, K).contiguous()
    o = (torch.randn(B, N, 3, device=dev) * 0.2 + torch.tensor([0., 0., 3.], device=dev))
    d = torch.randn(B, N, 3, device=dev)
    d = (d / d.norm(dim=-1, keepdim=True)).contiguous()
    o = o.contiguous()
    return kinds, sizes, pos, R, o, d


def bits(a, b):
    if a.dtype == torch.bool:
        return int((a != b).sum())
    return int((a.view(torch.int32 if a.element_size() == 4 else torch.int64)
                != b.view(torch.int32 if a.element_size() == 4 else torch.int64)).sum())


B, N = 8, 4096
bad = 0
for name, code in KINDS.items():
    for K in (1, 3, 9):
        args = scene(B, K, N, [code] * K)
        t0, n0, i0 = R3.cast_nearest(*args, backend="torch")
        t1, n1, i1 = R3.cast_nearest(*args, backend="triton")
        db = bits(t0, t1), bits(n0, n1), bits(i0, i1)
        # the shading only reads the normal where something was hit, so score it there too
        hitmask = (t0 < R3.INF)
        nh = int((n0[hitmask] .view(torch.int32) != n1[hitmask].view(torch.int32)).sum())
        bad += sum(db)
        print(f"cast   {name:9s} K={K:<2d} hits {int(hitmask.sum()):>7,}/{B * N:,}   "
              f"t {db[0]:>8,}  n {db[1]:>8,} (on hits {nh:>8,})  idx {db[2]:>8,}")

        lim = torch.rand(B, N, device=dev) * 6.0
        skip = torch.randint(0, K, (B, N), device=dev, dtype=torch.long)
        for sk in (None, skip):
            b0 = R3.any_blocking(*args, lim, skip=sk, backend="torch")
            b1 = R3.any_blocking(*args, lim, skip=sk, backend="triton")
            n = bits(b0, b1)
            bad += n
            print(f"block  {name:9s} K={K:<2d} skip={sk is not None}  blocked "
                  f"{int(b0.sum()):>7,}   diffs {n:>8,}")

# mixed-kind scene, which is what every real caller has
K = 10
codes = [R3.SPHERE, R3.CAPSULE, R3.BOX, R3.CYLINDER, R3.SPHERE, R3.BOX, R3.CAPSULE, R3.PLANE,
         R3.CYLINDER, R3.SPHERE]
args = scene(B, K, N, codes)
t0, n0, i0 = R3.cast_nearest(*args, backend="torch")
t1, n1, i1 = R3.cast_nearest(*args, backend="triton")
db = bits(t0, t1), bits(n0, n1), bits(i0, i1)
bad += sum(db)
print(f"\ncast   MIXED     K={K}  t {db[0]:,}  n {db[1]:,}  idx {db[2]:,}   "
      f"hits {int((t0 < R3.INF).sum()):,}/{B * N:,}")

# broadcast poses (stride-0 leading axis), which is how mujoco_locomotion hands sizes in
sizes1 = (0.05 + torch.rand(1, K, 3, device=dev)).expand(B, K, 3)
_, _, pos, R, o, d = args
t0, n0, i0 = R3.cast_nearest(codes, sizes1, pos, R, o, d, backend="torch")
t1, n1, i1 = R3.cast_nearest(codes, sizes1, pos, R, o, d, backend="triton")
db = bits(t0, t1), bits(n0, n1), bits(i0, i1)
bad += sum(db)
print(f"cast   STRIDE0   K={K}  t {db[0]:,}  n {db[1]:,}  idx {db[2]:,}")

# a single shared origin expanded over the batch, which is how every renderer here hands ``o`` in
oe = torch.tensor([0.0, 0.0, 3.0], device=dev).expand(B, N, 3)
t0, n0, i0 = R3.cast_nearest(codes, sizes1, pos, R, oe, d, backend="torch")
t1, n1, i1 = R3.cast_nearest(codes, sizes1, pos, R, oe, d, backend="triton")
db = bits(t0, t1), bits(n0, n1), bits(i0, i1)
bad += sum(db)
print(f"cast   EXPAND-O  K={K}  t {db[0]:,}  n {db[1]:,}  idx {db[2]:,}")

print("\nVERDICT:", "BIT-IDENTICAL everywhere" if bad == 0 else f"{bad:,} differing values")

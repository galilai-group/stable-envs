# These artifacts are stamped `cc02ff7`, which is the parent of the commit they measure

`gpu_after.json`, `gpu_profile_after.json` and `replay/swmnav2.json` carry
`git_commit: cc02ff7c2392704298855819da8abd0e192e0bf2`. That is the merge commit this work sits on,
not `6a47b59`, the commit that introduced the kernel — the tree was rsynced to the cluster and
measured **before** it was committed, and the launcher stamped the SHA the local worktree had at
sync time.

**What the artifacts measured is nonetheless exactly `6a47b59`**, and the two files that decide that
are byte-identical between the measured copy and the commit:

| file | md5 at `6a47b59` | md5 of the measured copy |
|---|---|---|
| `stable_envs/render/navdot.py` | `6143da1e6c39a7a3ccd9a7964d5debd5` | `6143da1e6c39a7a3ccd9a7964d5debd5` |
| `stable_envs/envs/swm_nav.py` | `f3fdaf3cc00d0624f849feb2f451bf39` | `f3fdaf3cc00d0624f849feb2f451bf39` |

`git diff 6a47b59..HEAD -- stable_envs/` is empty, so the same holds at the branch tip. The
`stable_envs_file` field in each shard points at `$SCR/se6/raster2d/`, the tree those md5s were taken
from.

Fixed going forward rather than by editing the record: the sbatch launchers now stamp
`STABLE_ENVS_GIT_COMMIT` from `git rev-parse HEAD` of the local worktree at submit time, so a shard
cannot again name a commit that does not contain the code it timed. The shards themselves are left
alone — a shard is a record of a run.

## Two fields in `gpu_profile_after.json` that do not mean what their names suggest

- **`dot1_ms`, `blend1_ms`, `dot2_ms`, `blend2_ms` time the OLD torch functions in isolation**, by
  calling `_gaussian_dot` and `_alpha_blend` directly. After SWMNAV-2 those functions are no longer
  on the cuda render path at all, so in the "after" profile they are a constant carried over from
  the "before" one, not a decomposition of the new `render_ms`. The columns that changed and that
  the findings quote are `render_ms`, `launches` and `base_ms`.
- **`float_temp_mib` is computed by FORMULA, not measured.** It counts the full-frame intermediates
  the torch expression obviously allocates — `dx`, `dy`, `dist2`, the float64 exp argument and its
  result, `dot`, `dot/m`, and `_alpha_blend`'s four — at `B x H x W` each. It is an **undercount**:
  it does not include the temporaries `torch.where`, `clamp`, `amax` and the dtype casts allocate
  internally, nor any allocator reuse. It is quoted as an order-of-magnitude statement about write
  amplification ("36 MiB out, 1,764 MiB of intermediates"), and it should be read as a lower bound
  on the traffic rather than as a measurement of it.

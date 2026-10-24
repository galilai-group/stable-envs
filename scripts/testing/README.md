# Testing scripts

Generators and verifiers for the sealed fixtures under `tests/fixtures/`.

| script | produces or checks |
|---|---|
| `freeze_native.py` | shared native trajectory corpus (`tests/fixtures/semantics/shared_native`) |
| `freeze_*_oracle.py` | per-family semantic oracles (`tests/fixtures/semantics/<family>`) |
| `freeze_occlusion_ball_oracle.py` | Blender G34 poses/depth/masks; isolated Python 3.11 + bpy 4.4 |
| `freeze_object_permanence_g62.py` | independent Blender G62 poses/depth/masks at all 120 source frames |
| `check_object_permanence_export.py` | live all-frame world-triangle/normal checks of production G34/G62 recordings |
| `_oracle.py` | pin check, rollout and manifest helpers the oracle generators share |
| `capture_ogbench_golden.py` | OGBench Cube/Scene golden frames |
| `verify_saved.py` | saved-corpus verification |

Run a script from the repository root, in the interpreter pinned for its family in
`tests/helpers/reference/profiles.py`:

```bash
python -m scripts.testing.freeze_reacher_oracle
```

Generators refuse to overwrite an existing directory. Move the old fixture aside, generate, then
compare before you replace it.

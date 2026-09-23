# PushT C3 exploratory evidence

- Reference checkout: `/Users/samib/stable-worldmodel`
- Reference commit: `176d54532b3c0a251a4f509b74f2fdacc143a63d`
- Reference tracked diff at start: clean (`git diff --quiet HEAD --` exit 0)
- Interpreter: `/private/tmp/stable-envs-semantic-recovery/.venv/bin/python`
- Platform: macOS 26.6.2 arm64
- Python: 3.12.11
- stable-worldmodel: 0.0.6
- Pymunk: 7.3.0
- NumPy: 2.5.3

Every physics probe used this prefix and saved raw stdout/stderr from execution:

```text
env SDL_VIDEODRIVER=dummy PYTHONPATH=/Users/samib/stable-worldmodel /private/tmp/stable-envs-semantic-recovery/.venv/bin/python SCRIPT 2> STDERR | tee STDOUT
```

Probe mapping:

- `01_cached_arbiter_restore.py`: body fields, then timestamp/current timestep/shape counter and exposed cached-arbiter restoration in a one-wall case.
- `02_cffi_schema.py`: confirms `cpBody` and `cpSpace` are opaque in the shipped CFFI schema while `cpArbiter` and `cpContact` fields are declared.
- `03_supported_space_copy.py`: Pymunk's supported `Space.copy()` continuation check. `03b_*` corresponds to the current script; the script was changed only to print source/copy vectors after the initial `03_*` run.
- `04_arbiter_roundtrip_membership.py`: exact exposed-arbiter dictionary roundtrip and body membership check.
- `05_live_contact_then_restore.py`: controlled wall contact, one detection step to reconstruct live membership, body restoration, then exposed-arbiter restoration.
- `06_wall_component_ablation.py`: wall-contact ablation of header, contact geometry/masses/bias/hash, and accumulated impulses.
- `07_agent_block_component_ablation.py`: three-step wall + agent-block contact ablation.
- `08_active_order.py`: semantic cached/body arbiter traversal order. `08_active_order.stdout.txt` is invalidated because the first script version restored block position before angle. `08b_active_order.stdout.txt` is the corrected angle-before-position result and corresponds to the current script.

Interpretation boundary:

These are exploratory, controlled CPU probes. They do not define or certify a public State. The wall-only case can be reconstructed exactly through supported stepping plus body restoration. The simultaneous wall and agent-block case remains non-exact after restoring every exposed body field, space scalar, live contact membership/order, and declared arbiter/contact field. Pymunk does not expose body bias velocities or the opaque `cpSpace` collision/active-solver bookkeeping through its Python API or shipped CFFI schema. Those private mechanisms remain untested.

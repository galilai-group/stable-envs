# Registry

Classes are canonical. The registry holds no semantics — only the class a name refers to, the SWM id aliases for it, and whether the class declared its fidelity tier completely before it was allowed in.

```python
import stable_envs as snv

snv.make("PushT", num_envs=8, device="cpu")
snv.make("swm/PushT-v1", num_envs=8, device="cpu")
snv.spec("PushT")
snv.list_envs()
snv.list_envs(family="classic_control")
snv.list_families()
snv.import_errors()
```

Registration is the `@register` decorator in `stable_envs.registry`. Duplicate class names are an error. A visual-tier class that omits measured `visual_fidelity` / `visual_note` / the words `VISUAL TIER` in its docstring does not import.

Auto-discovery of `stable_envs/envs/*` runs at `import stable_envs`. Modules whose name starts with `_` are skipped by discovery (`_craftax_full.py` is reached through the `craftax_full.py` alias). Import failures are recorded, never swallowed.

## SWM aliases

`snv.SWM_IDS` is the frozen Step-8 table, 42 rows. An id with several cells raises rather than picking a task. Symbolic ids raise rather than claiming a frame. See [environments](../environments.md).

## Step-8 names, until Wave 3

These still resolve and warn **once per process**:

```text
snv.StableEnv          # same object as snv.Env
snv.REGISTRY           # the private dict; prefer list_envs / spec / make
snv.register           # moved to stable_envs.registry.register
make(..., batch=…)     # spelling of num_envs=
```

They are removed in Wave 3, never before the Step-8 oracle comparison is green. New code should not use them. `dir(snv)` does not advertise them.

"""P6: what exactly makes griddly segfault at interpreter exit, and what stops it?

Four child processes, each building three griddly wrappers, rendering a frame and falling off the end
of the script so that teardown really runs. The exit CODE is the result (-11 is SIGSEGV).

  stock      griddly's own GymWrapper, bound to a script-level name
  noreload   a subclass whose __del__ does nothing, bound to a script-level name
  cached     stable_envs.envs.griddly._open() -- the module's process-wide cache
  stockcache griddly's own GymWrapper, but stashed in an imported module's dict

The pair (noreload, stock) isolates the Python finaliser; the pair (cached, stockcache) isolates where
the reference lives when the interpreter tears down.
"""
import os
import subprocess
import sys

CHILD = r'''
import sys, numpy as np
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
mode = sys.argv[1]
YAML = "Single-Player/GVGAI/sokoban.yaml"

if mode == "cached":
    import stable_envs.envs.griddly as G
    envs = [G._open(YAML, t) for t in (24, 32, 48)]
    for e in envs:
        e.reset(level_id=0)
        fr = np.asarray(e.game.observe())
    del envs                                   # only the module cache holds them now
else:
    from griddly import GymWrapper, gd
    if mode == "noreload":
        class W(GymWrapper):
            def __del__(self):
                pass
    else:
        W = GymWrapper
    made = [W(yaml_file=YAML, level=0,
              global_observer_type=gd.ObserverType.SPRITE_2D,
              player_observer_type=gd.ObserverType.SPRITE_2D) for _ in range(3)]
    for e in made:
        e.reset(level_id=0)
        fr = np.asarray(e.game.observe())
    if mode == "stockcache":
        import stable_envs.envs.griddly as G   # park them in an imported module's dict
        G._WRAPPERS.update({("probe", i): e for i, e in enumerate(made)})
        del made
print("rendered", fr.shape, len(np.unique(fr)), "distinct values", flush=True)
'''

if __name__ == "__main__":
    path = "/tmp/se_griddly_exit_child.py"
    with open(path, "w") as f:
        f.write(CHILD)
    for mode in ("stock", "noreload", "cached", "stockcache"):
        codes, out = [], ""
        for _ in range(6):
            p = subprocess.run([sys.executable, path, mode], capture_output=True, text=True)
            codes.append(p.returncode)
            out = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr.strip()[-200:]
        print(f"{mode:11s} exit codes over 6 runs: {codes}   last stdout: {out}", flush=True)
    sys.stdout.flush()

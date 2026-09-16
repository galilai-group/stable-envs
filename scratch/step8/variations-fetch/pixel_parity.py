"""Default Fetch frames from this branch and from the Step-8 baseline tree, on the same card.

The diff for this workstream is a new module plus one registration line in ``compat.worldmodel``,
and nothing on the render path imports either. "No pixel moved" is therefore structurally true, and
this measures it anyway: a structural argument is what people make right before they are wrong, and
64 frames on a card that is already allocated cost seconds.

Both trees are rendered in the SAME process would be wrong -- ``stable_envs`` is imported once -- so
this script renders ONE tree, selected by ``SE8_TREE_ROLE``, and the sbatch runs it twice. The
frames are written as ``.npy`` under the scratch root and compared by the second run.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

SCENES = ("Reach", "Push", "Slide", "PickAndPlace")
SIZE = 128
N = 8
OUT = os.path.join("examples", "results", "step8", "variations-fetch")


def render(tree: str) -> dict:
    """Render the four scenes' default frames in a FRESH interpreter rooted at ``tree``."""
    code = (
        "import os,sys,json,numpy as np,torch;"
        f"sys.path.insert(0,{tree!r});"
        "import stable_envs as se;"
        "out={};"
        f"scenes={list(SCENES)!r};"
        "\nfor s in scenes:\n"
        f"    e=se.make('Fetch'+s,batch={N},size={SIZE},device='cuda')\n"
        f"    st=e.sample_states({N},rng=np.random.default_rng(0))\n"
        "    f=e.render(st).cpu().numpy()\n"
        f"    np.save(os.path.join(os.environ['SE8_FRAMES'],s+'.npy'),f)\n"
        "    out[s]=[int(f.sum()),list(f.shape)]\n"
        "print('JSON'+json.dumps(out))")
    env = dict(os.environ)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise SystemExit(f"render in {tree} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    line = [x for x in r.stdout.splitlines() if x.startswith("JSON")][-1]
    return json.loads(line[4:])


def main() -> int:
    base = os.environ["BASE_TREE"]
    here = os.getcwd()
    root = os.path.join(os.environ.get("SCR", "/tmp"), "se8_frames")
    res = {}
    for role, tree in (("branch", here), ("baseline", base)):
        d = os.path.join(root, role)
        os.makedirs(d, exist_ok=True)
        os.environ["SE8_FRAMES"] = d
        res[role] = render(tree)
    cmp = {}
    ok = True
    for s in SCENES:
        a = np.load(os.path.join(root, "branch", s + ".npy"))
        b = np.load(os.path.join(root, "baseline", s + ".npy"))
        d = np.abs(a.astype(np.int32) - b.astype(np.int32))
        cmp[s] = dict(maxdiff=int(d.max()), frac_changed=float((d.max(-1) > 0).mean()),
                      shape=list(a.shape))
        ok &= int(d.max()) == 0
    out = dict(baseline_tree=base, branch_tree=here, size=SIZE, n_states=N,
               node=os.uname().nodename, slurm_job_id=os.environ.get("SLURM_JOB_ID"),
               branch_sha=os.environ.get("STABLE_ENVS_GIT_COMMIT"),
               per_scene=cmp, verdict="IDENTICAL" if ok else "PIXELS_MOVED")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "pixel_parity.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

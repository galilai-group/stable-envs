"""Replay the Step-4 FROZEN Fetch corpora through this branch and through the Step-8 baseline tree.

``pixel_parity.py`` next door samples eight states per scene with ``sample_states``. That is a smoke
check and the independent review was right to say so: merge-gate item 5 asks for the *affected
frozen-corpus replay*, and the corpora that are affected are the ones Step 4 froze and digested ---
``examples/results/step4/states/Fetch*.npz``, 64 states per configuration, whose sha256 and whose
frame digest are both recorded in ``examples/results/step4/conformance/fetch.json``.

So this loads those exact arrays, renders them at the size Step 4 rendered them (480, supersample 1),
and makes three comparisons:

  1. **branch vs baseline, byte for byte.** Both trees in the same job on the same card, the same
     corpus bytes handed to both. This is the claim "this workstream moved no pixel", measured.
  2. **branch vs the frozen Step-4 digest.** ``oracle.digest_ours`` from the committed shard. This is
     the claim "the frozen oracle still replays", and it is the one that would catch a drift that
     happened to affect both trees equally.
  3. **the corpus itself.** sha256 of the state array as loaded, against the ``corpus.sha256`` Step 4
     wrote, and against the baseline tree's own copy of the same file. A replay whose input silently
     changed proves nothing.

The corpus is read from ONE tree (the branch) and handed to both renderers, so a difference can only
come from the renderer. The baseline tree's copy is hashed separately and asserted equal rather than
assumed.

``stable_envs`` can only be imported once per interpreter, so, exactly as in ``pixel_parity.py``, one
tree is rendered per subprocess and this script drives two of them.

Run inside an allocation, e.g.::

    srun --jobid=<alloc> --overlap -n 1 --cpus-per-task=6 --gres=gpu:1 \
        bash scratch/step8/variations-fetch/frozen_replay.sh
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import numpy as np

#: The four render-distinct Fetch scenes. The other eight ids are the ``Dense``/``Dict`` siblings,
#: which Step 4 measured to draw a byte-identical picture (``sibling_digest_matches``) and whose
#: frozen corpora are the same arrays; the sibling corpus sha256s are checked below rather than
#: re-rendered, because re-rendering them would measure the same kernel three times.
SCENES = ("FetchReach", "FetchPush", "FetchSlide", "FetchPickAndPlace")
SIBLINGS = {s: (s + "Dense", s + "Dict") for s in SCENES}

SIZE = 480
OUT = os.path.join("examples", "results", "step8", "variations-fetch")
SHARD = os.path.join("examples", "results", "step4", "conformance", "fetch.json")


def _sha(a: np.ndarray) -> str:
    """``conformance.frames_digest``'s recipe, reimplemented so the comparison is like for like.

    Step 4 hashed its corpora with ``frames_digest``, which feeds ``shape`` and ``dtype`` into the
    hash before the bytes. A plain ``sha256(tobytes())`` here would differ from every recorded
    ``corpus.sha256`` and the mismatch would look like a corpus that had moved. Spelled out rather
    than imported because this driver runs outside either tree's ``stable_envs``.
    """
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _file_sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_corpus(tree: str, cfg: str) -> tuple[np.ndarray, dict]:
    p = os.path.join(tree, "examples", "results", "step4", "states", cfg + ".npz")
    z = np.load(p, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return np.ascontiguousarray(np.asarray(z["states"], np.float32)), meta


def render(tree: str, frames_dir: str) -> dict:
    """Render every scene's frozen corpus in a FRESH interpreter rooted at ``tree``.

    The corpus arrays are passed on disk (``$SE8_CORPUS/<cfg>.npy``) rather than re-loaded from the
    tree being rendered, so that both renderers provably see the same bytes.
    """
    code = (
        "import os,sys,json,hashlib,numpy as np;"
        f"sys.path.insert(0,{tree!r});"
        "import stable_envs as se;"
        "from stable_envs.testing import conformance as C;"
        "out={};"
        f"scenes={list(SCENES)!r};"
        "\nfor s in scenes:\n"
        "    st=np.load(os.path.join(os.environ['SE8_CORPUS'],s+'.npy'))\n"
        f"    e=se.make(s,batch=len(st),size={SIZE},device='cuda')\n"
        "    f=e.render(st).cpu().numpy()\n"
        "    np.save(os.path.join(os.environ['SE8_FRAMES'],s+'.npy'),f)\n"
        "    ds=np.tile(np.asarray(e.default_style(),np.float32),(len(st),1))\n"
        "    out[s]={'digest_ours':C.frames_digest(f),"
        "            'digest_ours_default_style':C.frames_digest(e.render(st,style=ds)),"
        "            'frames_shape':list(f.shape),'mean_pixel':float(f.mean()),"
        "            'state_dim':int(e.state_dim),'style_dim':int(e.style_dim),"
        "            'size':int(e.size),'supersample':int(e.ss),'device':str(e.device),"
        "            'states_sha256':C.frames_digest(np.ascontiguousarray(st))}\n"
        "import mujoco,mujoco_warp,torch\n"
        "out['_versions']={'mujoco':mujoco.__version__,'mujoco_warp':mujoco_warp.__version__,"
        "                  'torch':torch.__version__,'numpy':np.__version__,"
        "                  'python':sys.version.split()[0],'gpu':torch.cuda.get_device_name(0)}\n"
        "print('JSON'+json.dumps(out))")
    env = dict(os.environ, SE8_FRAMES=frames_dir)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise SystemExit(f"replay in {tree} failed:\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}")
    line = [x for x in r.stdout.splitlines() if x.startswith("JSON")][-1]
    return json.loads(line[4:])


def main() -> int:
    here = os.getcwd()
    base = os.environ["BASE_TREE"]
    root = os.path.join(os.environ.get("SCR", "/tmp"), "se8_replay")
    corpus_dir = os.path.join(root, "corpus")
    os.makedirs(corpus_dir, exist_ok=True)

    shard = json.load(open(os.path.join(here, SHARD)))["configs"]

    # ---- 1. the corpus, from the branch tree, hashed, and the baseline's copy asserted equal
    corpora = {}
    for cfg in SCENES:
        states, meta = load_corpus(here, cfg)
        np.save(os.path.join(corpus_dir, cfg + ".npy"), states)
        rec = shard[cfg]["corpus"]
        entry = {
            "n_states": int(states.shape[0]), "state_dim": int(states.shape[1]),
            "states_sha256": _sha(states),
            "step4_recorded_sha256": rec.get("sha256"),
            "npz_file_sha256": {
                "branch": _file_sha(os.path.join(here, "examples/results/step4/states", cfg + ".npz")),
                "baseline": _file_sha(os.path.join(base, "examples/results/step4/states", cfg + ".npz")),
            },
            "selection": rec.get("selection"), "pool": rec.get("pool"), "seed": rec.get("seed"),
            "loaded_from": f"examples/results/step4/states/{cfg}.npz",
        }
        entry["matches_step4_record"] = entry["states_sha256"] == entry["step4_recorded_sha256"]
        entry["identical_file_in_both_trees"] = (
            entry["npz_file_sha256"]["branch"] == entry["npz_file_sha256"]["baseline"])
        # the two siblings ship the same frozen states; hashed, not re-rendered
        entry["siblings"] = {}
        for sib in SIBLINGS[cfg]:
            s_states, _ = load_corpus(here, sib)
            entry["siblings"][sib] = {
                "states_sha256": _sha(s_states),
                "identical_to_scene_corpus": _sha(s_states) == entry["states_sha256"],
                "step4_digest_ours": (shard[sib].get("oracle") or {}).get("digest_ours"),
                "step4_digest_matches_scene": ((shard[sib].get("oracle") or {}).get("digest_ours")
                                               == (shard[cfg].get("oracle") or {}).get("digest_ours")),
            }
        corpora[cfg] = entry

    # ---- 2. render both trees, same card, same job, same corpus bytes
    os.environ["SE8_CORPUS"] = corpus_dir
    runs = {}
    for role, tree in (("branch", here), ("baseline", base)):
        d = os.path.join(root, role)
        os.makedirs(d, exist_ok=True)
        runs[role] = render(tree, d)

    # ---- 3. compare
    per_scene, ok, oracle_ok = {}, True, True
    for cfg in SCENES:
        a = np.load(os.path.join(root, "branch", cfg + ".npy"))
        b = np.load(os.path.join(root, "baseline", cfg + ".npy"))
        d = np.abs(a.astype(np.int32) - b.astype(np.int32))
        step4 = (shard[cfg].get("oracle") or {})
        row = {
            "frames_shape": list(a.shape),
            "branch_vs_baseline": {
                "maxdiff": int(d.max()), "frac_changed": float((d.max(-1) > 0).mean()),
                "bytes_identical": bool(np.array_equal(a, b)),
            },
            "digest_ours": {"branch": runs["branch"][cfg]["digest_ours"],
                            "baseline": runs["baseline"][cfg]["digest_ours"],
                            "step4_frozen": step4.get("digest_ours")},
            "digest_ours_default_style": {
                "branch": runs["branch"][cfg]["digest_ours_default_style"],
                "baseline": runs["baseline"][cfg]["digest_ours_default_style"],
                "step4_frozen": step4.get("digest_ours_default_style")},
            "mean_pixel": {"branch": runs["branch"][cfg]["mean_pixel"],
                           "baseline": runs["baseline"][cfg]["mean_pixel"]},
            "states_sha256_seen_by_renderer": {
                "branch": runs["branch"][cfg]["states_sha256"],
                "baseline": runs["baseline"][cfg]["states_sha256"]},
            "step4_render_config": {k: step4.get(k) for k in
                                    ("size", "supersample", "batch", "n_states", "device")},
        }
        row["branch_matches_baseline"] = (row["branch_vs_baseline"]["bytes_identical"]
                                          and row["digest_ours"]["branch"] == row["digest_ours"]["baseline"])
        row["branch_matches_step4_frozen_digest"] = (
            row["digest_ours"]["branch"] == row["digest_ours"]["step4_frozen"])
        row["baseline_matches_step4_frozen_digest"] = (
            row["digest_ours"]["baseline"] == row["digest_ours"]["step4_frozen"])
        ok &= bool(row["branch_matches_baseline"])
        oracle_ok &= bool(row["branch_matches_step4_frozen_digest"])
        per_scene[cfg] = row

    verdict = ("IDENTICAL_AND_ORACLE_REPLAYS" if ok and oracle_ok else
               "IDENTICAL_TO_BASELINE_ORACLE_DRIFTED" if ok else "PIXELS_MOVED")
    out = {
        "what_this_is": ("merge-gate item 5 for Workstream B: the Step-4 frozen Fetch corpora "
                         "replayed through this branch and through the Step-8 baseline tree in one "
                         "job on one card, compared byte for byte and against the frozen Step-4 "
                         "frame digests"),
        "provenance": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "host": os.uname().nodename,
            "branch_sha": os.environ.get("STABLE_ENVS_GIT_COMMIT"),
            "baseline_sha": os.environ.get("SE8_BASELINE_SHA"),
            "branch_tree": here.replace(os.environ.get("SCR", "\0"), "$SCR"),
            "baseline_tree": base.replace(os.environ.get("SCR", "\0"), "$SCR"),
            "venv": os.environ.get("SE8_VENV"),
            "size": SIZE,
            **runs["branch"].get("_versions", {}),
            "baseline_versions": runs["baseline"].get("_versions", {}),
        },
        "corpora": corpora,
        "per_scene": per_scene,
        "verdict": verdict,
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "frozen_corpus_replay.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=False)
    print(json.dumps(out, indent=1))
    return 0 if ok and oracle_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

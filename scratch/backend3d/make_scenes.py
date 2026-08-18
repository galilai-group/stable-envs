"""Dump self-contained MJCF scenes used as benchmark fixtures for the 3-D backend study.

Runs in the `test` conda env (which has ogbench); writes plain XML that any env with
plain `mujoco` can load, so the benchmark venvs never need ogbench.
"""
import os, shutil, sys, xml.etree.ElementTree as ET

OUT = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/backend3d/scenes"
os.makedirs(OUT, exist_ok=True)


def copy_assets(xml_path, search_roots):
    """mj_saveLastXML emits bare `file=` references; carry the referenced assets along
    so the scene loads in an env that has no ogbench installed."""
    tree = ET.parse(xml_path)
    index = {}
    for root in search_roots:
        for dirpath, _, files in os.walk(root):
            for f in files:
                index.setdefault(f, os.path.join(dirpath, f))
    missing = []
    for el in tree.iter():
        f = el.get("file")
        if not f:
            continue
        base = os.path.basename(f)
        dst = os.path.join(os.path.dirname(xml_path), base)
        src = index.get(base)
        if src and not os.path.exists(dst):
            shutil.copy(src, dst)
        elif not src:
            missing.append(base)
        el.set("file", base)
    for el in tree.iter("compiler"):
        for attr in ("meshdir", "texturedir", "assetdir"):
            el.attrib.pop(attr, None)
    tree.write(xml_path)
    if missing:
        print(f"  missing assets: {sorted(set(missing))}")


def dump(env_name, fname):
    import ogbench, mujoco
    env = ogbench.make_env_and_datasets(env_name, env_only=True)
    m = env.unwrapped.model
    path = os.path.join(OUT, fname)
    mujoco.mj_saveLastXML(path, m)
    copy_assets(path, [os.path.dirname(ogbench.__file__)])
    print(f"{env_name} -> {path}  ngeom={m.ngeom} nmesh={m.nmesh} nq={m.nq} ncam={m.ncam}")
    return path


if __name__ == "__main__":
    for env_name, fname in [
        ("antmaze-medium-navigate-v0", "antmaze_medium.xml"),
        ("pointmaze-medium-navigate-v0", "pointmaze_medium.xml"),
        ("humanoidmaze-medium-navigate-v0", "humanoidmaze_medium.xml"),
        ("cube-single-play-v0", "cube_single.xml"),
    ]:
        try:
            dump(env_name, fname)
        except Exception as e:
            print(f"FAILED {env_name}: {type(e).__name__}: {e}")

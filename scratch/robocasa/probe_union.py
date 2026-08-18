"""Make the compiled model a PER-KITCHEN constant instead of per-episode content.

G3 fails for RoboCasa because ``reset()`` recompiles the scene. But the two halves of that scene
behave completely differently, and ``probe_bucket.py`` measured the split: at a pinned
(layout, style) the KITCHEN contributes exactly 1,185 geoms on every one of 24 resets, while the
sampled OBJECTS contribute 13-85. The whole per-episode delta is three free bodies.

So the question is not "does the model change" (it does) but "does it have to". A model that holds
the kitchen plus the UNION of every object any episode in that bucket uses is compiled once; each
episode is then rendered by placing its own three objects from qpos and parking the rest below the
floor, which is a pure qpos operation and therefore batchable. This probe builds that model out of
the corpus XMLs and asks the only question that matters for a repo whose contract is parity:

    does the union model render each episode BIT-IDENTICALLY to that episode's own model?

It also prices the union: compile time, geom/mesh/texture counts and .mjb size as a function of how
many episodes are folded in, which is what says whether the approach survives a real bucket size.
"""
from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import os
import time
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
PARK = np.array([0.0, 0.0, -50.0, 1.0, 0.0, 0.0, 0.0])     # far below the floor, out of every view

REF_ATTRS = ("mesh", "material", "texture", "hfield", "joint", "body", "site", "class",
             "childclass", "target", "objname")


def sha(s):
    return hashlib.sha1(s.encode() if isinstance(s, str) else s).hexdigest()[:16]


def asset_key(el: ET.Element) -> str:
    """Identity of an asset by CONTENT, so the same object drawn twice is stored once."""
    return el.tag + "|" + "|".join(f"{k}={v}" for k, v in sorted(el.attrib.items()) if k != "name")


def collect_refs(body: ET.Element) -> set[str]:
    out = set()
    for el in body.iter():
        for a in ("mesh", "material", "hfield"):
            if el.get(a):
                out.add(el.get(a))
    return out


def names_in(body: ET.Element) -> set[str]:
    return {el.get("name") for el in body.iter() if el.get("name")}


def rename(el: ET.Element, mapping: dict[str, str]) -> None:
    for e in el.iter():
        n = e.get("name")
        if n is not None and n in mapping:
            e.set("name", mapping[n])
        for a in REF_ATTRS:
            v = e.get(a)
            if v is not None and v in mapping:
                e.set(a, mapping[v])


class UnionBuilder:
    """Fold object bodies from many episode MJCFs into one model, sharing identical assets."""

    def __init__(self, base_xml: str, obj_roots: list[str]):
        self.tree = ET.ElementTree(ET.fromstring(base_xml))
        self.root = self.tree.getroot()
        self.wb = self.root.find("worldbody")
        self.asset = self.root.find("asset")
        self.by_key = {asset_key(e): e.get("name") for e in self.asset}
        self.taken = {e.get("name") for e in self.root.iter() if e.get("name")}
        self.slots: dict[str, list[str]] = {}                 # episode tag -> parked body names
        self.slots["ep000"] = list(obj_roots)
        self.n_assets_added = self.n_assets_shared = 0

    def add_episode(self, tag: str, xml: str, obj_roots: list[str]) -> list[str]:
        r = ET.fromstring(xml)
        wb, asset = r.find("worldbody"), r.find("asset")
        # RoboCasa gives a texture and the material that uses it the SAME name, so an asset table
        # keyed by name alone silently loses one of the two -- which shows up much later as
        # "texture X not found in material X" on a model that looked fine to build.
        by_kind: dict[str, dict[str, ET.Element]] = {"mesh": {}, "material": {}, "texture": {}}
        for e in asset:
            if e.tag in by_kind:
                by_kind[e.tag][e.get("name")] = e
        added = []
        for root_name in obj_roots:
            body = next((b for b in wb.findall("body") if b.get("name") == root_name), None)
            if body is None:
                continue
            body = copy.deepcopy(body)
            # 1. assets this body needs, transitively: geom -> mesh/material, material -> texture
            want = {"mesh": set(), "material": set(), "texture": set()}
            for el in body.iter():
                for a in ("mesh", "material"):
                    if el.get(a) and el.get(a) in by_kind[a]:
                        want[a].add(el.get(a))
            for nm in want["material"]:
                t = by_kind["material"][nm].get("texture")
                if t and t in by_kind["texture"]:
                    want["texture"].add(t)
            # 2. add them, sharing anything byte-identical that is already in the union
            amap = {"mesh": {}, "material": {}, "texture": {}}
            for kind in ("mesh", "texture", "material"):       # texture before material
                for nm in sorted(want[kind]):
                    el = copy.deepcopy(by_kind[kind][nm])
                    if kind == "material" and el.get("texture") in amap["texture"]:
                        el.set("texture", amap["texture"][el.get("texture")])
                    k = asset_key(el)
                    if k in self.by_key:
                        amap[kind][nm] = self.by_key[k]
                        self.n_assets_shared += 1
                        continue
                    new = f"{tag}_{nm}"
                    while new in self.taken:
                        new += "_"
                    amap[kind][nm] = new
                    el.set("name", new)
                    self.asset.append(el)
                    self.taken.add(new)
                    self.by_key[k] = new
                    self.n_assets_added += 1
            # 3. rename the body's own names, then repoint mesh/material references by KIND
            bmap = {n: f"{tag}_{n}" for n in names_in(body)}
            for el in body.iter():
                n = el.get("name")
                if n is not None and n in bmap:
                    el.set("name", bmap[n])
                for a in ("mesh", "material"):
                    v = el.get(a)
                    if v is not None and v in amap[a]:
                        el.set(a, amap[a][v])
            self.wb.append(body)
            added.append(bmap[root_name])
        self.slots[tag] = added
        return added

    def to_string(self) -> str:
        return ET.tostring(self.root, encoding="unicode")


def eff_alpha(m) -> np.ndarray:
    a = m.geom_rgba[:, 3].copy()
    hm = m.geom_matid >= 0
    a[hm] = m.mat_rgba[m.geom_matid[hm], 3]
    return a


def names_of(m, kind):
    adr = {"body": m.name_bodyadr, "geom": m.name_geomadr, "jnt": m.name_jntadr,
           "mesh": m.name_meshadr}[kind]
    raw = bytes(m.names)
    return [raw[a:raw.index(b"\x00", a)].decode() for a in adr]


def subtree_mask(m, roots):
    bn = names_of(m, "body")
    idx = {n: i for i, n in enumerate(bn)}
    rs = {idx[n] for n in roots if n in idx}
    inside = np.zeros(m.nbody, bool)
    for b in range(m.nbody):
        p, k = b, 0
        while p != 0 and k < 64:
            if p in rs:
                inside[b] = True
                break
            p = int(m.body_parentid[p]); k += 1
    return inside[m.geom_bodyid]


def geom_sig(m, mask):
    gn, mn = np.array(names_of(m, "geom")), np.array(names_of(m, "mesh") or [""])
    rows = []
    for g in np.nonzero(mask)[0]:
        mesh = mn[m.geom_dataid[g]] if (m.geom_type[g] == 7 and m.geom_dataid[g] >= 0) else ""
        rows.append("|".join([gn[g], str(int(m.geom_type[g])), str(int(m.geom_group[g])), mesh,
                              np.array2string(m.geom_size[g], precision=6),
                              np.array2string(m.geom_pos[g], precision=6),
                              np.array2string(m.geom_quat[g], precision=6),
                              np.array2string(m.geom_rgba[g], precision=6)]))
    return sha("\n".join(sorted(rows)))


def render_group1(mujoco, m, d, size, cam):
    r = mujoco.Renderer(m, height=size, width=size)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[1] = 1
    r.update_scene(d, camera=cam, scene_option=opt)
    img = r.render().copy()
    r.close()
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(OUT, "corpus"))
    ap.add_argument("--kitchens", type=int, default=4)
    ap.add_argument("--eps", type=int, default=16)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--cam", default="robot0_agentview_left")
    ap.add_argument("--out", default=os.path.join(OUT, "union.json"))
    args = ap.parse_args()

    import mujoco
    summary = json.load(open(os.path.join(args.corpus, "summary.json")))
    rec = {"mujoco": mujoco.__version__, "res": args.res, "kitchens": []}

    for kinfo in summary["kitchens"][:args.kitchens]:
        tag = kinfo["tag"]
        d = os.path.join(args.corpus, tag)
        eps = sorted(glob.glob(os.path.join(d, "ep*.xml")))[:args.eps]
        K = {"tag": tag, "n_eps": len(eps), "episodes": []}
        print(f"\n===== {tag}: {len(eps)} episodes =====", flush=True)

        # ---- is the DRAWN kitchen a constant of (layout, style)? --------------------------------
        per = []
        for i, x in enumerate(eps):
            m = mujoco.MjModel.from_xml_path(x)
            roots = kinfo["episodes"][i]["obj_roots"]
            om = subtree_mask(m, roots)
            a = eff_alpha(m)
            vis = (a > 0) & (m.geom_group == 1)
            per.append(dict(ngeom=int(m.ngeom), nmesh=int(m.nmesh), ntex=int(m.ntex),
                            n_obj_geoms=int(om.sum()),
                            kit_all=geom_sig(m, ~om),
                            kit_vis=geom_sig(m, (~om) & vis),
                            kit_vis_n=int(((~om) & vis).sum()),
                            kit_invis_n=int(((~om) & (m.geom_group == 1) & (a <= 0)).sum()),
                            obj_vis_n=int((om & vis).sum())))
            del m
        K["distinct_kitchen_all"] = len({p["kit_all"] for p in per})
        K["distinct_kitchen_visible"] = len({p["kit_vis"] for p in per})
        K["kitchen_visible_geoms"] = sorted({p["kit_vis_n"] for p in per})
        K["kitchen_invisible_geoms"] = sorted({p["kit_invis_n"] for p in per})
        K["object_visible_geoms"] = sorted({p["obj_vis_n"] for p in per})
        print("distinct kitchen sig  all:", K["distinct_kitchen_all"],
              " visible-only:", K["distinct_kitchen_visible"],
              " visible kitchen geoms:", K["kitchen_visible_geoms"],
              " invisible:", K["kitchen_invisible_geoms"], flush=True)

        # ---- build the union ------------------------------------------------------------------
        base_xml = open(eps[0]).read()
        ub = UnionBuilder(base_xml, kinfo["episodes"][0]["obj_roots"])
        growth = []
        for i, x in enumerate(eps[1:], start=1):
            ub.add_episode(f"e{i:03d}", open(x).read(), kinfo["episodes"][i]["obj_roots"])
            if (i + 1) % 4 == 0 or i == len(eps) - 1:
                s = ub.to_string()
                t0 = time.perf_counter()
                mu = mujoco.MjModel.from_xml_string(s)
                growth.append(dict(n_eps=i + 1, compile_s=time.perf_counter() - t0,
                                   ngeom=int(mu.ngeom), nmesh=int(mu.nmesh), ntex=int(mu.ntex),
                                   nq=int(mu.nq), nmeshface=int(mu.nmeshface),
                                   texdata_MB=float(mu.tex_data.size / 1e6),
                                   xml_MB=len(s) / 1e6,
                                   assets_added=ub.n_assets_added, assets_shared=ub.n_assets_shared))
                print("  union@%d eps: ngeom=%d nmesh=%d ntex=%d nq=%d compile=%.2fs tex=%.0fMB"
                      % (i + 1, mu.ngeom, mu.nmesh, mu.ntex, mu.nq, growth[-1]["compile_s"],
                         growth[-1]["texdata_MB"]), flush=True)
                del mu
        K["growth"] = growth

        union_xml = ub.to_string()
        with open(os.path.join(d, "union.xml"), "w") as f:
            f.write(union_xml)
        mu = mujoco.MjModel.from_xml_string(union_xml)
        du = mujoco.MjData(mu)
        mjb = os.path.join(d, "union.mjb")
        mujoco.mj_saveModel(mu, mjb, None)
        K["union"] = dict(ngeom=int(mu.ngeom), nq=int(mu.nq), nmesh=int(mu.nmesh),
                          ntex=int(mu.ntex), nmeshface=int(mu.nmeshface),
                          texdata_MB=float(mu.tex_data.size / 1e6),
                          mjb_MB=os.path.getsize(mjb) / 1e6, xml_MB=len(union_xml) / 1e6)
        print("  UNION:", json.dumps(K["union"]), flush=True)

        # ---- does it render each episode bit-identically? --------------------------------------
        ujn = names_of(mu, "jnt")
        uadr = {n: (int(mu.jnt_qposadr[i]), int(mu.jnt_type[i])) for i, n in enumerate(ujn)}
        width = {0: 7, 1: 4, 2: 1, 3: 1}
        for i, x in enumerate(eps):
            me = mujoco.MjModel.from_xml_path(x)
            de = mujoco.MjData(me)
            q = np.load(x.replace(".xml", "_qpos.npy"))
            de.qpos[:] = q
            mujoco.mj_forward(me, de)
            ref = render_group1(mujoco, me, de, args.res, args.cam)

            # union: park every free body, then place this episode's joints by name
            qu = np.array(mu.qpos0, dtype=np.float64).copy()
            for j, n in enumerate(ujn):
                if int(mu.jnt_type[j]) == 0:
                    qu[int(mu.jnt_qposadr[j]):int(mu.jnt_qposadr[j]) + 7] = PARK
            ejn = names_of(me, "jnt")
            pref = "" if i == 0 else f"e{i:03d}_"
            mapped = miss = 0
            for j, n in enumerate(ejn):
                cand = n if n in uadr else pref + n
                if cand not in uadr:
                    miss += 1
                    continue
                a, _ = uadr[cand]
                w = width[int(me.jnt_type[j])]
                b = int(me.jnt_qposadr[j])
                qu[a:a + w] = q[b:b + w]
                mapped += 1
            du.qpos[:] = qu
            mujoco.mj_forward(mu, du)
            got = render_group1(mujoco, mu, du, args.res, args.cam)
            dd = np.abs(got.astype(np.int16) - ref.astype(np.int16))
            row = dict(ep=os.path.basename(x), mapped=mapped, unmapped=miss,
                       px_exact=float((dd.max(-1) == 0).mean()), px_mean=float(dd.mean()),
                       px_max=int(dd.max()), px_within8=float((dd.max(-1) <= 8).mean()))
            K["episodes"].append(row)
            print("  ", json.dumps(row), flush=True)
            if i == 0:
                try:
                    import imageio.v2 as imageio
                    imageio.imwrite(os.path.join(d, "union_ref.png"), ref)
                    imageio.imwrite(os.path.join(d, "union_got.png"), got)
                except Exception:
                    pass
            del me, de
        K["all_exact"] = all(e["px_exact"] == 1.0 for e in K["episodes"])
        K["min_px_exact"] = min(e["px_exact"] for e in K["episodes"])
        print(f"  ALL BIT-EXACT: {K['all_exact']}   worst exact fraction {K['min_px_exact']:.6f}",
              flush=True)
        rec["kitchens"].append(K)
        del mu, du

    with open(args.out, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

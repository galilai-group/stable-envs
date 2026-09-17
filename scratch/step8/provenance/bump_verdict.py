"""The mujoco / mujoco-warp 3.13 bump verdict, assembled from the committed measurement arms.

    python scratch/step8/provenance/bump_verdict.py

writes ``examples/results/step8/provenance/mjwarp313_verdict.json``.

Plan section 14 sets three conditions and a bump is accepted only if ALL hold:

    1. it fixes the relevant defect
    2. existing fidelity/regression evidence is rechecked
    3. it does not silently move unrelated renderers

This file states each condition, the artifact that answers it, and the answer. It reads; it does
not measure. The measurements are ``examples/results/step8/provenance/mjwarp313/*.json`` (jobs run
under SLURM allocation 6443480 on gpu2264, one RTX 3090) and the committed Step-7 shards they are
compared against.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
M = ROOT / "examples/results/step8/provenance/mjwarp313"
OUT = ROOT / "examples/results/step8/provenance/mjwarp313_verdict.json"
OGB_REF = ROOT / "examples/results/step7/closure2/gate_a/cuda_ogbmanip.json"


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def build() -> dict:
    tex312 = _load(M / "texgap_se7-ogb.json")
    tex313 = _load(M / "texgap_ogb_313.json")
    d312 = _load(M / "digest_se7-ogb.json")
    d313 = _load(M / "digest_se8-mjwarp313.json")
    dogb = _load(M / "digest_ogb_313.json")
    dfetch = _load(M / "digest_se-s3b-fetch311.json")
    r312 = _load(M / "replay_dmc_se7-ogb.json")["configs"]["DMCCheetahRun"]
    r313 = _load(M / "replay_dmc_se8-mjwarp313.json")["configs"]["DMCCheetahRun"]
    ogbref = _load(OGB_REF)

    def fam(row, check, *keys):
        ev = ((row.get("family_specific") or {}).get(check) or {}).get("evidence") or {}
        return {k: ev.get(k) for k in keys}

    digests = {
        "OGBCube": {
            "declared_reference": "mujoco 3.12.0, ogbench 1.2.1",
            "3.12": (ogbref["configs"]["OGBCube"].get("oracle") or {}).get("digest_ours"),
            "3.12_source": str(OGB_REF.relative_to(ROOT)),
            "3.13": dogb["cells"]["OGBCube"].get("digest_ours"),
            "3.13_source": str((M / "digest_ogb_313.json").relative_to(ROOT))},
        "OGBScene": {
            "declared_reference": "mujoco 3.12.0, ogbench 1.2.1",
            "3.12": (ogbref["configs"]["OGBScene"].get("oracle") or {}).get("digest_ours"),
            "3.12_source": str(OGB_REF.relative_to(ROOT)),
            "3.13": dogb["cells"]["OGBScene"].get("digest_ours"),
            "3.13_source": str((M / "digest_ogb_313.json").relative_to(ROOT))},
        "DMCCheetahRun": {
            "declared_reference": "mujoco 3.11.0",
            "3.12": d312["cells"]["DMCCheetahRun"].get("digest_ours"),
            "3.12_source": str((M / "digest_se7-ogb.json").relative_to(ROOT)),
            "3.13": d313["cells"]["DMCCheetahRun"].get("digest_ours"),
            "3.13_source": str((M / "digest_se8-mjwarp313.json").relative_to(ROOT)),
            "caveat": "this cell's oracle digest is documented as NOT run-to-run reproducible "
                      "(mujoco_warp's batched render; examples/results/step6/texture-fidelity/"
                      "findings.md S5), so the digest alone is not the evidence here -- the "
                      "fidelity and downsample numbers below are"},
        "FetchReach": {
            "declared_reference": "mujoco 3.11.0 (the Step-3B Fetch pin)",
            "3.11": dfetch["cells"]["FetchReach"].get("digest_ours"),
            "3.11_source": str((M / "digest_se-s3b-fetch311.json").relative_to(ROOT)),
            "3.12": d312["cells"]["FetchReach"].get("why"),
            "3.13": d313["cells"]["FetchReach"].get("why"),
            "why": "gymnasium-robotics' set_joint_qpos assertion fires under mujoco >= 3.12, so the "
                   "Fetch REFERENCE env cannot be built at all on 3.12 or 3.13. This is upstream "
                   "and predates 3.13 (docs/research/swm_matrix.md); it is recorded here because a "
                   "bump would make it the shipped configuration"},
    }
    for v in digests.values():
        a, b = v.get("3.12"), v.get("3.13")
        v["byte_identical"] = (a == b) if (a and b and len(str(a)) == 64 and len(str(b)) == 64) else None

    conditions = {
        "1_fixes_the_defect": {
            "answer": "YES",
            "evidence": {
                "mujoco_warp 3.12.0 sample_texture guards a missing texcoord":
                    tex312["sample_texture_guards_missing_texcoords"],
                "mujoco_warp 3.13.0 sample_texture guards a missing texcoord":
                    tex313["sample_texture_guards_missing_texcoords"],
                "sample_texture sha256 3.12": tex312["sample_texture_sha256"],
                "sample_texture sha256 3.13": tex313["sample_texture_sha256"],
                "upstream fix": "the MESH branch now reads texcoord_offset = "
                                "mesh_texcoord_offsets[mesh_id] and skips the lookup when it is "
                                "negative, leaving uv at (0,0). The comment names mjm.mesh_texcoordadr "
                                "== -1 -- exactly the condition Step-7 W5 diagnosed on OGBScene's two "
                                "stopbuttonrim geoms (examples/results/step7/ogb-manipulation/"
                                "findings.md S11)",
                "OGBScene still reports the gap under 3.13": tex313.get("texture_gaps"),
                "why that is not a contradiction":
                    "stable_envs.render.mjwarp.texture_gaps describes the MODEL (a textured mesh "
                    "with no UVs), not the renderer's tolerance of it. The model has not changed, so "
                    "the report has not. What changed is that mujoco_warp no longer dies on it"},
        },
        "2_existing_evidence_rechecked": {
            "answer": "YES, and it does not hold",
            "evidence": {
                "DMCCheetahRun check_visual_fidelity 3.12":
                    fam(r312, "check_visual_fidelity", "mean_abs", "within8_frac", "ssim"),
                "DMCCheetahRun check_visual_fidelity 3.13":
                    fam(r313, "check_visual_fidelity", "mean_abs", "within8_frac", "ssim"),
                "DMCCheetahRun check_resolution_consistency 3.12":
                    fam(r312, "check_resolution_consistency", "exact_frac"),
                "DMCCheetahRun check_resolution_consistency 3.13":
                    fam(r313, "check_resolution_consistency", "exact_frac"),
                "frozen bound": "tests/gpu/test_reproducibility.py:DOWNSAMPLE_EXACT_FRAC"
                                "[DMCCheetahRun] = 0.533850, margin 0.005, floor 0.528850",
                "consequence": "0.533854 under 3.12 (the bound, to 4e-6) -> 0.522233 under 3.13, "
                               "which is BELOW the floor. A 3.13 bump turns a passing family_specific "
                               "Gate-A check into a FAIL on a cell the texcoord defect never touched",
            },
        },
        "3_moves_nothing_unrelated": {
            "answer": "NO",
            "evidence": {
                "digests": digests,
                "render.py changes beyond the texcoord guard (source diff, 3.12 -> 3.13)": [
                    "vertex_normal(): a vertex normal more than ~37 degrees off its face is replaced "
                    "by the face normal (matching mjr_uploadMesh) -- a SHADING change on every mesh",
                    "BVH traversal widened by RAY_TOL_REL / RAY_TOL_ABS and coincident hits now "
                    "tie-break to the higher geom index 'like MuJoCo's GL depth test' -- a VISIBILITY "
                    "change on every scene with touching surfaces",
                    "the no-light fallback is now a shadow_light_fraction parameter rather than the "
                    "NO_LIGHT_AMBIENT_FALLBACK constant -- a LIGHTING change",
                    "a new _aa_resolve kernel accumulates and averages samples -- a SAMPLING change",
                    "render() restructured into _build_megakernel()",
                ],
                "reading": "each of these is plausibly a correctness improvement upstream. None of "
                           "them is the texcoord fix, and every one of them can move a pixel in a "
                           "cell that has no textured mesh at all. That is what condition 3 asks "
                           "about, and the digests confirm it in the frames",
            },
        },
    }

    return {
        "what_this_is": "the mujoco / mujoco-warp 3.13.0 bump evaluation against plan section 14's "
                        "three conditions, assembled from committed measurement arms. No pin is "
                        "changed by this workstream; this is a recommendation.",
        "generated_at_head_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                         text=True).strip(),
        "candidate": {"mujoco": "3.13.0", "mujoco-warp": "3.13.0"},
        "control": {"venv": "$SCR/venvs/se8-mjwarp313",
                    "built_from": "$SCR/venvs/se7-ogb's pin set with ONLY the MuJoCo pair bumped",
                    "other_differences": ["cuda_bindings 12.9.7 -> 12.9.8",
                                          "cuda_pathfinder 1.6.0 -> 1.8.1",
                                          "filelock 3.32.3 -> 3.32.7",
                                          "setuptools 78.1.0 -> 84.0.0",
                                          "urllib3 2.7.0 -> 2.8.0",
                                          "stable-worldmodel 0.1.1 added --no-deps for the OGB arm"],
                    "hardware": "SLURM allocation 6443480, gpu2264, NVIDIA GeForce RTX 3090, "
                                "driver 575.57.08, MUJOCO_GL=egl",
                    "scripts": ["scratch/step8/provenance/build_mjwarp313.sh",
                                "scratch/step8/provenance/probe313.sh",
                                "scratch/step8/provenance/probe313b.sh",
                                "scratch/step8/provenance/probe313c.sh",
                                "scratch/step8/provenance/texgap_probe.py",
                                "scratch/step8/provenance/digest_replay.py"]},
        "conditions": conditions,
        "recommendation": "DO NOT BUMP",
        "recommendation_why":
            "Condition 1 holds and conditions 2 and 3 do not. mujoco_warp 3.13.0 does fix the "
            "UV-less-mesh crash -- the guard is in the shipped source and names the exact field -- "
            "but the same release also changes ray tie-breaking, vertex normals, the no-light "
            "fallback and sample resolution, and every frozen digest measured here moves: OGBCube, "
            "OGBScene and DMCCheetahRun. DMCCheetahRun's downsample number crosses its frozen floor, "
            "so the bump would create a Gate-A FAIL. And the bump buys nothing that is currently "
            "costing anything: OGBScene already renders, because the cell bakes the unsamplable "
            "material away, at a DECLARED residual of 0.139 % of pixels with interior_exact 1.0. "
            "Dropping that bake under 3.13 would itself change pixels and need a re-declaration. "
            "Separately, mujoco >= 3.12 cannot build the Fetch reference env at all, so 3.13 is not "
            "a candidate for the interpreter that certifies Fetch either.",
        "what_would_change_the_answer":
            "a release that carries the texcoord guard and NOTHING else in the render path, or a "
            "deliberate re-declaration campaign that re-measures every mjwarp cell under the new "
            "pair and re-freezes with the drift documented. Both are Step-9 scope. Until then the "
            "finding to record upstream is that the fix and the renderer changes shipped together.",
        "arms": {str(p.relative_to(ROOT)): _sha256(p) for p in sorted(M.glob("*.json"))},
        "compared_against": {str(OGB_REF.relative_to(ROOT)): _sha256(OGB_REF)},
    }


if __name__ == "__main__":
    doc = build()
    OUT.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: doc["conditions"][k]["answer"] for k in doc["conditions"]}, indent=1))
    print("recommendation:", doc["recommendation"])

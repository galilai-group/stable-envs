"""Stage C — put the two renders side by side so the quality can be JUDGED rather than asserted.

Three panels, same trajectory, same states: the official MuJoCo/robosuite raster, the MJWarp raster,
and 4x the absolute difference. The third panel is the point. A side-by-side alone invites the eye to
declare two images "basically the same"; an amplified residual says exactly where they are not, and on
this scene it draws the texgen gap for you -- the textured boxes MJWarp cannot UV light up flat while
the mesh silhouettes stay dark.

Also written, because a video is not always the artifact a reader can open: a GIF, a still contact
sheet at evenly spaced frames, and the per-frame numbers as JSON.

CODEC. ``cv2.VideoWriter`` with the ``mp4v`` fourcc writes MPEG-4 Part 2, which no browser plays in an
HTML5 ``<video>`` element -- the page shows a blank player with controls, which reads as a broken build
rather than as a codec choice. Encoding goes through ffmpeg with ``-c:v libx264 -pix_fmt yuv420p
-profile:v baseline -movflags +faststart``, and the resulting stream is PROBED and the probe recorded,
so "it is H.264" is a measurement.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess

import cv2
import numpy as np

DEFAULT_OUT = "/mnt/vast/home/randall.balestriero/robocasa_visual_work"
LABEL_H = 22
GAP = 6
BG = 245


# ------------------------------------------------------------------ metrics


def interior_mask(frames: np.ndarray) -> np.ndarray:
    """The repo's own definition: pixels whose 3x3 neighbourhood in the REFERENCE is one flat colour.

    Copied in behaviour from ``stable_envs.testing.conformance.interior_mask`` rather than imported,
    because this script runs in the robocasa conda env where ``stable_envs`` is not installed. The
    output is ``(N, H-1, W-1)``, matching that function, so the numbers here and the numbers the
    conformance suite prints are the same quantity.
    """
    import scipy.ndimage as ndi
    out = []
    for f in frames:
        g = f.astype(np.int16)
        flat = (np.abs(np.diff(g, axis=0)).sum(-1)[:, :-1]
                + np.abs(np.diff(g, axis=1)).sum(-1)[:-1]) == 0
        out.append(ndi.minimum_filter(flat, size=3))
    return np.stack(out)


def ssim(a: np.ndarray, b: np.ndarray, win: int = 7) -> float:
    """Mean SSIM over a uint8 stack, on luminance, with the standard 8-bit constants.

    Written out rather than imported: ``skimage`` is not in this environment, and SSIM on a
    uniform-filter window is eight lines. Luminance rather than per-channel because the residual here
    is a colour-and-texture error that shows in luma, and a per-channel mean would triple-count it.
    """
    import scipy.ndimage as ndi
    f = lambda x: ndi.uniform_filter(x, size=win, mode="nearest")
    x = (a.astype(np.float64) @ [0.299, 0.587, 0.114])
    y = (b.astype(np.float64) @ [0.299, 0.587, 0.114])
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ux, uy = f(x), f(y)
    vx, vy = f(x * x) - ux * ux, f(y * y) - uy * uy
    vxy = f(x * y) - ux * uy
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux ** 2 + uy ** 2 + c1) * (vx + vy + c2))
    return float(s.mean())


def fidelity(official: np.ndarray, ours: np.ndarray) -> dict:
    """Every number the visual tier declares, plus the per-frame series the video prints."""
    d = np.abs(official.astype(np.int16) - ours.astype(np.int16)).max(-1)
    im = interior_mask(official)
    di = d[:, :-1, :-1]
    inter = di[im]
    per = [{"mean_abs": float(x.mean()), "within8": float((x <= 8).mean()),
            "exact": float((x == 0).mean())} for x in d]
    return {
        "frames": int(len(d)),
        "mean_abs": float(d.mean()), "within8_frac": float((d <= 8).mean()),
        "within16_frac": float((d <= 16).mean()), "within32_frac": float((d <= 32).mean()),
        "exact_frac": float((d == 0).mean()), "max_abs": int(d.max()),
        "p95_abs": float(np.percentile(d, 95)),
        "interior_frac": float(im.mean()),
        "interior_mean_abs": float(inter.mean()) if inter.size else None,
        "interior_exact_frac": float((inter == 0).mean()) if inter.size else None,
        "interior_within8_frac": float((inter <= 8).mean()) if inter.size else None,
        "ssim": ssim(official, ours),
        "per_frame": per,
    }


# ------------------------------------------------------------------ composition


def label(img: np.ndarray, text: str) -> np.ndarray:
    """Caption in a band ABOVE the frame, never on top of it.

    These images are the evidence for a fidelity claim, so nothing drawn by this script may be
    mistakable for something a renderer produced.
    """
    band = np.full((LABEL_H, img.shape[1], 3), BG, np.uint8)
    scale = 0.40
    while scale > 0.18:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        if tw <= img.shape[1] - 6:
            break
        scale -= 0.02
    cv2.putText(band, text, (3, LABEL_H - 7), cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 1,
                cv2.LINE_AA)
    return np.concatenate([band, img], 0)


def upscale(f: np.ndarray, k: int) -> np.ndarray:
    """Nearest-neighbour only. Smoothing would invent detail neither renderer produced."""
    return f if k <= 1 else np.repeat(np.repeat(f, k, axis=-3), k, axis=-2)


def even(v: np.ndarray) -> np.ndarray:
    """yuv420p needs even dimensions; pad with the gutter colour rather than cropping a frame."""
    if v.shape[1] % 2:
        v = np.concatenate([v, np.full((len(v), 1, v.shape[2], 3), BG, np.uint8)], 1)
    if v.shape[2] % 2:
        v = np.concatenate([v, np.full((len(v), v.shape[1], 1, 3), BG, np.uint8)], 2)
    return v


def compose(official, ours, fid, kitchen, camera, amp=4, k=2):
    """(T,H,W,3) x2 -> (T,H',W',3) triptych with labels and a per-frame footer."""
    t = len(official)
    diff = np.clip(np.abs(official.astype(np.int16) - ours.astype(np.int16)) * amp, 0, 255
                   ).astype(np.uint8)
    o, m, d = (upscale(x, k) for x in (official, ours, diff))
    h, w = o.shape[1:3]
    out = []
    for i in range(t):
        row = np.concatenate([
            label(o[i], "official: MuJoCo EGL (robosuite, geom group 1)"),
            np.full((h + LABEL_H, GAP, 3), BG, np.uint8),
            label(m[i], "MJWarp batch raytracer (visual tier)"),
            np.full((h + LABEL_H, GAP, 3), BG, np.uint8),
            label(d[i], f"|difference| x{amp}"),
        ], 1)
        foot = np.full((2 * LABEL_H, row.shape[1], 3), BG, np.uint8)
        p = fid["per_frame"][i]
        cv2.putText(foot, f"{kitchen}  {camera}  frame {i+1}/{t}", (6, LABEL_H - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(foot, f"this frame: mean |diff| {p['mean_abs']:.2f}/255   "
                          f"within 8/255 {100*p['within8']:.1f}%   "
                          f"whole clip SSIM {fid['ssim']:.4f}", (6, 2 * LABEL_H - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
        out.append(np.concatenate([row, foot], 0))
    return even(np.stack(out))


def contact_sheet(official, ours, fid, kitchen, camera, n=6, amp=4, k=2):
    """Rows official / MJWarp / 4x|diff| at evenly spaced frames — the video as one still."""
    idx = np.linspace(0, len(official) - 1, n).astype(int)
    diff = np.clip(np.abs(official.astype(np.int16) - ours.astype(np.int16)) * amp, 0, 255
                   ).astype(np.uint8)
    rows = []
    for name, stack in (("official MuJoCo EGL", official), ("MJWarp visual tier", ours),
                        (f"|difference| x{amp}", diff)):
        cells = [label(upscale(stack[i], k), f"{name}  t={i}") for i in idx]
        rows.append(np.concatenate([np.concatenate([c, np.full((c.shape[0], GAP, 3), BG, np.uint8)], 1)
                                    for c in cells], 1))
    sheet = np.concatenate([np.concatenate([r, np.full((GAP, r.shape[1], 3), BG, np.uint8)], 0)
                            for r in rows], 0)
    cap = np.full((2 * LABEL_H, sheet.shape[1], 3), BG, np.uint8)
    cv2.putText(cap, f"RoboCasa {kitchen} / {camera}: same qpos, two renderers", (6, LABEL_H - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(cap, f"mean |diff| {fid['mean_abs']:.2f}/255, {100*fid['within8_frac']:.1f}% within "
                     f"8/255, SSIM {fid['ssim']:.4f}, interior exact "
                     f"{100*(fid['interior_exact_frac'] or 0):.1f}%", (6, 2 * LABEL_H - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (20, 20, 20), 1, cv2.LINE_AA)
    return np.concatenate([cap, sheet], 0)


def tile(frames, cols, labels=None, pad=3, k=1):
    f = upscale(frames, k)
    cells = [label(x, labels[i]) if labels else x for i, x in enumerate(f)]
    h, w = cells[0].shape[:2]
    rows = math.ceil(len(cells) / cols)
    canvas = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad, 3), BG, np.uint8)
    for i, c in enumerate(cells):
        r, cc = divmod(i, cols)
        canvas[pad + r * (h + pad):pad + r * (h + pad) + h,
               pad + cc * (w + pad):pad + cc * (w + pad) + w] = c
    return canvas


# ------------------------------------------------------------------ encoding


def probe_codec(path: str) -> str:
    """The stream line, from ffprobe if present, else parsed out of ffmpeg's own report."""
    import imageio_ffmpeg
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        p = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=codec_name,profile,codec_tag_string", "-of", "default=nw=1",
                            path], capture_output=True, text=True)
        if p.returncode == 0 and p.stdout.strip():
            return " ".join(p.stdout.split())
    p = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", path],
                       capture_output=True, text=True)
    for line in p.stderr.splitlines():
        if "Stream #" in line and "Video:" in line:
            return line.strip()
    return "unknown"


def write_h264(path: str, frames: np.ndarray, fps: int = 20) -> dict:
    import imageio_ffmpeg
    t, h, w, _ = frames.shape
    assert h % 2 == 0 and w % 2 == 0, f"yuv420p needs even dimensions, got {h}x{w}"
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "baseline",
           "-movflags", "+faststart", "-crf", "17", path]
    p = subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {p.stderr.decode()[-2000:]}")
    codec = probe_codec(path)
    assert "h264" in codec, f"{path} is not H.264 -- browsers will show a blank player: {codec}"
    return {"path": path, "bytes": os.path.getsize(path), "frames": int(t), "fps": fps,
            "size": [int(h), int(w)], "codec": codec}


def write_gif(path: str, frames: np.ndarray, fps: int = 10, max_frames: int = 80) -> dict:
    import imageio.v2 as imageio
    step = max(1, len(frames) // max_frames)
    sub = list(frames[::step])
    imageio.mimsave(path, sub, duration=1.0 / fps, loop=0)
    return {"path": path, "bytes": os.path.getsize(path), "frames": len(sub)}


def png(path: str, img: np.ndarray) -> str:
    cv2.imwrite(path, img[:, :, ::-1])
    return path


# ------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitchens", nargs="+", default=["L52S03"])
    ap.add_argument("--camera", default="robot0_agentview_left")
    ap.add_argument("--work", default=DEFAULT_OUT)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "media"))
    ap.add_argument("--upscale", type=int, default=2)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    manifest = {"kitchens": {}, "grids": {}}

    for kitchen in args.kitchens:
        d = os.path.join(args.work, kitchen)
        meta = json.load(open(os.path.join(d, "meta.json")))
        entry = {"meta": {k: meta.get(k) for k in
                          ("task", "layout", "style", "steps", "res", "seed", "reset_s",
                           "official_render_fps", "official_restore_render_fps", "mujoco")},
                 "cameras": {}}
        entry["model"] = {k: meta["model"][k] for k in ("nq", "ngeom", "nmesh", "ntex", "nmeshface")}
        for cam in meta["cameras"]:
            o = np.load(os.path.join(d, f"official_{cam}.npy"))
            w = np.load(os.path.join(d, f"mjwarp_{cam}.npy"))
            fid = fidelity(o, w)
            entry["cameras"][cam] = {k: v for k, v in fid.items() if k != "per_frame"}
            # The flat-mean bake of the textured primitives MJWarp cannot UV, scored against the same
            # official frames. It is an ABLATION, not the shipped path: on this scene it does not pay,
            # and a mitigation that is not scored against the thing it mitigates is a decoration.
            bp = os.path.join(d, f"mjwarpbake_{cam}.npy")
            if os.path.exists(bp):
                bf = fidelity(o, np.load(bp))
                entry["cameras"][cam]["ablation_primitive_texture_bake"] = {
                    k: bf[k] for k in ("mean_abs", "within8_frac", "exact_frac",
                                       "interior_mean_abs", "interior_exact_frac", "ssim")}
            if cam != args.camera:
                continue
            vid = compose(o, w, fid, kitchen, cam, k=args.upscale)
            entry["video"] = write_h264(os.path.join(args.out, f"sidebyside_{kitchen}.mp4"),
                                        vid, fps=args.fps)
            # Half size and every fourth frame: a GIF of photographic content is 5x the mp4 for the
            # same clip, and this one exists only for viewers that will not run a <video> element.
            entry["gif"] = write_gif(os.path.join(args.out, f"sidebyside_{kitchen}.gif"),
                                     vid[:, ::2, ::2], max_frames=50)
            entry["contact_sheet"] = png(
                os.path.join(args.out, f"contact_{kitchen}.png"),
                contact_sheet(o, w, fid, kitchen, cam, k=args.upscale))
        manifest["kitchens"][kitchen] = entry
        print(f"{kitchen}: " + json.dumps(
            {c: {k: round(v, 4) for k, v in s.items() if isinstance(v, float)}
             for c, s in entry["cameras"].items()}, indent=1), flush=True)

    labels = {}
    lp = os.path.join(args.work, "grid_labels.json")
    if os.path.exists(lp):
        labels = json.load(open(lp))
    g = os.path.join(args.work, "dr_one_batch.npy")
    if os.path.exists(g):
        f = np.load(g)
        manifest["grids"]["dr_one_batch"] = png(
            os.path.join(args.out, "dr_one_batch.png"),
            tile(f, 4, [f"world {i}: camera + light randomised" if i else "world 0: official"
                        for i in range(len(f))], k=args.upscale))
    g = os.path.join(args.work, "kitchen_grid.npy")
    if os.path.exists(g):
        f = np.load(g)
        manifest["grids"]["kitchen_grid"] = png(
            os.path.join(args.out, "kitchen_grid.png"),
            tile(f, 4, labels.get("kitchen_grid"), k=args.upscale))

    for name in os.listdir(args.work):
        if name.startswith("mjwarp_") and name.endswith(".json"):
            manifest.setdefault("mjwarp", {})[name] = json.load(open(os.path.join(args.work, name)))
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("wrote", os.path.join(args.out, "manifest.json"), flush=True)


if __name__ == "__main__":
    main()

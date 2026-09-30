"""Test exact saved Bullet render poses instead of inferred velocity lag."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .collect import candidate
from .common import catalog, load_state, rgb, state_hash, write
from .config_fixes import mse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("campaign", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--visual-pose", choices=["cached", "current"], default="cached")
    p.add_argument("--bullet-camera", action="store_true")
    p.add_argument("--bullet-mvp", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    from stable_envs.envs import rocket_landing as frozen
    from stable_envs.render import nvdiff as ND

    row = catalog()["swm/PFRocketLanding-v0"]
    corpus = a.campaign / "run-6750648_7/results/PyFlytRocketLanding/cpu"
    meta = json.loads((corpus / "result.json").read_text())
    env, active_config = None, object()
    result = dict(status="RUNNING", visual_pose=a.visual_pose, frames=[])
    for f in meta["frames"]:
        config = meta["recipes"][f["case"]]["simulator_config"]
        if config != active_config:
            if env is not None:
                env.close()
            env = candidate(row, "cuda", 0, config)
            c = env._nvd(480, 1)
            active_config = config
        state = load_state(corpus / f["state_file"], "cuda", env.get_state())
        assert state_hash(state) == f["state_hash"]
        host = state.cpu()
        links = host["current", "diagnostic", "cached_links"].numpy()
        pos = np.zeros((1, 13, 3), np.float64)
        quat = np.zeros((1, 13, 4), np.float64)
        quat[..., 3] = 1
        for geom, body in [(0, 0), (1, 1), (12, 2)]:
            pos[:, geom] = host["current", "physics", f"body_{body}", "position"].numpy()
            quat[:, geom] = host["current", "physics", f"body_{body}", "quaternion"].numpy()
        pos[:, 2:12] = links[:, :, :3]
        quat[:, 2:12] = links[:, :, 3:7]
        if a.visual_pose == "current":
            local_pos, local_quat = frozen.link_local_frames()
            rotation = frozen.quat_to_mat(quat[:, 1])
            for j in range(1, len(local_pos)):
                pos[:, 1 + j] = pos[:, 1] + np.einsum("bij,j->bi", rotation, local_pos[j])
                quat[:, 1 + j] = frozen.quat_mul(quat[:, 1], local_quat[j : j + 1])
        # Rebase before float32 upload. At the high-ceiling checkpoints, absolute
        # world coordinates otherwise lose millimetres/centimetres of detail.
        world_pos = pos.copy()
        origin = pos[:, 1].copy()
        pos -= origin[:, None]
        eye, cam = frozen.camera_pose(links[:, 0, :3] - origin, links[:, 0, 3:7])
        camera = ND.Camera(eye, cam, frozen.CAM_FOV, frozen.CAM_NEAR, frozen.CAM_FAR)
        if a.bullet_camera:
            import pybullet

            # Preserve Bullet's world-coordinate float32 look-at arithmetic,
            # then rebase its resulting matrix without changing that camera.
            eye_world, camera_world = frozen.camera_pose(links[:, 0, :3], links[:, 0, 3:7])
            views = []
            for i in range(len(eye)):
                q = links[i, 0, 3:7]
                r = np.asarray(pybullet.getMatrixFromQuaternion(q)).reshape(3, 3)
                e = links[i, 0, :3] + r @ frozen.CAM_OFFSET
                angles = np.asarray(pybullet.getEulerFromQuaternion(q))
                angles[1] += np.pi / 6
                up = np.asarray(pybullet.getMatrixFromQuaternion(pybullet.getQuaternionFromEuler(angles))).reshape(
                    3, 3
                )[:, 2]
                view = np.asarray(pybullet.computeViewMatrix(e, links[i, 0, :3], up)).reshape(4, 4).T.copy()
                view[:3, 3] += view[:3, :3] @ origin[i]
                views.append(view)
            projection = (
                np.asarray(pybullet.computeProjectionMatrixFOV(frozen.CAM_FOV, 1.0, frozen.CAM_NEAR, frozen.CAM_FAR))
                .reshape(4, 4)
                .T.copy()
            )
            camera.matrices = lambda aspect: (np.stack(views), projection)
            if a.bullet_mvp:
                # TinyRenderer first forms view*model, then projection*(view*model),
                # with reverse-order float32 dot products. At 500 km the rounding
                # is visibly different from a mathematically equivalent double MVP.
                def mm(x, y):
                    out = np.zeros(np.broadcast_shapes(x.shape[:-2], y.shape[:-2]) + (4, 4), np.float32)
                    for k in (3, 2, 1, 0):
                        out += x[..., :, k : k + 1] * y[..., k : k + 1, :]
                    return out

                model = np.zeros((1, 13, 4, 4), np.float32)
                model[..., :3, :3] = frozen.quat_to_mat(quat)
                model[..., :3, 3] = world_pos
                model[..., 3, 3] = 1
                world_view = np.asarray(pybullet.computeViewMatrix(e, links[0, 0, :3], up), np.float32).reshape(4, 4).T
                mvp = torch.as_tensor(mm(projection.astype(np.float32), mm(world_view, model)), device="cuda")
                scene = c.br.scene
                original_clip = ND.Scene.clip.__get__(scene)

                def clip(xpos, xmat, view, proj):
                    _, normals, world = original_clip(xpos, xmat, view, proj)
                    positions = torch.sparse.mm(scene.S4, mvp.permute(1, 3, 0, 2).reshape(4 * scene.G, 4))
                    return positions.reshape(scene.V, 1, 4).permute(1, 0, 2).contiguous(), normals, world

                scene.clip = clip
        rs = env.to_render_state(state)
        colors = frozen.material_table(rs["renderer", "style"].cpu().numpy())
        image = rgb(
            c.br.render(
                torch.tensor(pos, dtype=torch.float32, device="cuda"),
                torch.tensor(frozen.quat_to_mat(quat), dtype=torch.float32, device="cuda"),
                camera,
                mat_rgb=torch.tensor(colors, dtype=torch.float32, device="cuda"),
            )
        ).copy()
        source = np.load(corpus / f["oracle_file"], allow_pickle=False)
        np.savez_compressed(a.output / f"frame-{f['index']:04d}.npz", ours=image, source=source)
        result["frames"].append(
            dict(
                index=f["index"],
                case=f["case"],
                seed=f["seed"],
                step=f["step"],
                state_hash=f["state_hash"],
                mse=mse(image, source),
            )
        )
        write(a.output / "result.json", result)
        print(f["index"], f["step"], result["frames"][-1]["mse"], flush=True)
    result["status"] = "COMPLETE"
    write(a.output / "result.json", result)


if __name__ == "__main__":
    main()

"""Research: draw saved numeric Rocket poses in a private TinyRenderer world.

No physics stepping, replay, source renderer invocation or image substitution.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .collect import candidate
from .common import catalog, load_state, state_hash, timed, write
from .config_fixes import mse


def camera_matrices(link, bullet):
    rotation = np.array(bullet.getMatrixFromQuaternion(link[3:7])).reshape(3, 3)
    eye = link[:3] + rotation @ np.array([-1.0, 0.0, 3.0])
    angles = np.array(bullet.getEulerFromQuaternion(link[3:7]))
    angles[1] += np.pi / 6
    up = np.array(bullet.getMatrixFromQuaternion(bullet.getQuaternionFromEuler(angles))).reshape(3, 3)[:, 2]
    return bullet.computeViewMatrix(eye, link[:3], up), bullet.computeProjectionMatrixFOV(90.0, 1.0, 0.1, 255.0)


def main():
    import pybullet as bullet
    import pybullet_data
    import PyFlyt

    p = argparse.ArgumentParser()
    p.add_argument("campaign", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    client = bullet.connect(bullet.DIRECT)
    assets = Path(PyFlyt.__file__).parent
    ground = bullet.loadURDF(
        str(Path(pybullet_data.getDataPath()) / "plane.urdf"), useFixedBase=True, physicsClientId=client
    )
    rocket = bullet.loadURDF(
        str(assets / "models/vehicles/rocket/rocket.urdf"),
        flags=bullet.URDF_USE_INERTIA_FROM_FILE,
        physicsClientId=client,
    )
    pad = bullet.loadURDF(str(assets / "models/landing_pad.urdf"), physicsClientId=client)
    link_names = {
        bullet.getJointInfo(rocket, i, physicsClientId=client)[12].decode(): i
        for i in range(bullet.getNumJoints(rocket, physicsClientId=client))
    }
    print("links", link_names, flush=True)
    row = catalog()["swm/PFRocketLanding-v0"]
    corpus = a.campaign / "run-6750648_7/results/PyFlytRocketLanding/cpu"
    meta = json.loads((corpus / "result.json").read_text())
    env, config = None, object()
    result = dict(status="RUNNING", frames=[])
    try:
        for f in meta["frames"]:
            next_config = meta["recipes"][f["case"]]["simulator_config"]
            if next_config != config:
                if env is not None:
                    env.close()
                env = candidate(row, "cpu", 0, next_config)
                config = next_config
            state = load_state(corpus / f["state_file"], "cpu", env.get_state())
            assert state_hash(state) == f["state_hash"]
            rgba = state["render", "rgba"][0].numpy()
            for idx, body in enumerate((ground, rocket, pad)):
                position = state["current", "physics", f"body_{idx}", "position"][0].numpy()
                quaternion = state["current", "physics", f"body_{idx}", "quaternion"][0].numpy()
                bullet.resetBasePositionAndOrientation(body, position, quaternion, physicsClientId=client)
            bullet.changeVisualShape(rocket, -1, rgbaColor=rgba[0], physicsClientId=client)
            bullet.changeVisualShape(pad, -1, rgbaColor=rgba[4], physicsClientId=client)
            for name, idx in link_names.items():
                color = (
                    rgba[1]
                    if "fin" in name
                    else rgba[2]
                    if "leg" in name
                    else rgba[3]
                    if "booster" in name
                    else rgba[5]
                    if "flame" in name
                    else None
                )
                if color is not None:
                    bullet.changeVisualShape(rocket, idx, rgbaColor=color, physicsClientId=client)
            view, projection = camera_matrices(state["current", "diagnostic", "cached_links"][0, 0].numpy(), bullet)

            def draw():
                raw = bullet.getCameraImage(
                    480,
                    480,
                    viewMatrix=view,
                    projectionMatrix=projection,
                    renderer=bullet.ER_TINY_RENDERER,
                    physicsClientId=client,
                )[2]
                return np.asarray(raw, np.uint8).reshape(480, 480, 4)[..., :3].copy()

            timing, image = timed(draw, samples=1, warmups=1)
            source = np.load(corpus / f["oracle_file"], allow_pickle=False)
            np.savez_compressed(a.output / f"frame-{f['index']:04d}.npz", ours=image, source=source)
            result["frames"].append(
                dict(index=f["index"], state_hash=f["state_hash"], mse=mse(image, source), timing=timing)
            )
            assert state_hash(state) == f["state_hash"]
            write(a.output / "result.json", result)
            print(f["index"], result["frames"][-1]["mse"], flush=True)
        result["status"] = "COMPLETE"
        write(a.output / "result.json", result)
    finally:
        bullet.disconnect(client)
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()

import numpy as np

from stable_worldmodel.envs.pusht.env import PushT
from stable_worldmodel.envs.quantizer import PolarQuantizer


options = {
    "variation": [],
    "state": np.array([100, 100, 400, 400, 0, 0, 0], np.float64),
    "goal_state": np.array([200, 200, 300, 300, 0, 0, 0], np.float64),
}
for action in [
    np.array([0, 0], np.float32),
    np.array([0.25, -0.5], np.float64),
    np.array([2, 0], np.float32),
]:
    env = PushT()
    observation, _ = env.reset(seed=0, options=options)
    initial = observation["state"].copy()
    target = np.array(env.agent.position) + action * 100
    output = env.step(action)
    print(
        "continuous",
        action.dtype,
        action.tolist(),
        "initial",
        initial.tolist(),
        "derived_target",
        target.tolist(),
        "next",
        output[0]["state"].tolist(),
        "rtdt",
        output[1:4],
    )
    env.close()

quantizer = PolarQuantizer(16, 16, 1.0)
for action in [
    np.array([0, 0]),
    np.array([0, 4]),
    np.array([8, 8]),
    np.array([15, 15]),
    np.array([-1, 16]),
]:
    print(
        "discrete",
        action.tolist(),
        "contains?",
        bool(np.all((action >= 0) & (action < 16))),
        "dequant",
        quantizer.dequantize(action).tolist(),
    )

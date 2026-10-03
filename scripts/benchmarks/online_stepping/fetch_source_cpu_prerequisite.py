"""Direct pinned Fetch source versus accepted adapters; no GL or timing.

All 12 registrations independently cover float32/64 actions, B2, owned full
State, different-history restoration and unseeded reset RNG. This is not public
renderer, process transport, throughput or source-render lifecycle acceptance.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import mujoco
import numpy as np
import stable_worldmodel
import stable_worldmodel.envs  # noqa: F401
import torch
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

from scripts.benchmarks.online_stepping.fetch_source_render_lifecycle import IDS, PIN
from scripts.recovery.fetch_public_gate import equal_state, exact_array, full_compare
from stable_envs._adapters.fetch_pickandplace import FetchAdapter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = Path(stable_worldmodel.__file__).resolve().parents[1]
    manifest = json.loads(args.source_manifest.read_text())
    assert manifest['commit'] == PIN and mujoco.__version__ == '3.11.0'
    for name, digest in manifest['files'].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest, name
    torch.set_num_threads(1)
    files = [Path(__file__).relative_to(Path.cwd()),
             Path('scripts/benchmarks/online_stepping/fetch_source_render_lifecycle.py'),
             Path('scripts/recovery/fetch_public_gate.py'), Path('scripts/recovery/fetch_pickandplace_adapter_probe.py'),
             Path('stable_envs/_adapters/fetch_pickandplace.py'), Path('stable_envs/_adapters/fetch_observations.py')]
    report = dict(status='RUNNING', source_pin=PIN, source_files=manifest['files'], host=platform.node(),
        versions={n: importlib.metadata.version(n) for n in
                  ('numpy', 'mujoco', 'gymnasium', 'gymnasium-robotics', 'torch', 'tensordict')},
        scope='CPU B2 direct actual source and real adapters, no public Env/GL/process/timing acceptance',
        no_rendering=True, no_timing=True,
        candidate_files={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}, cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    save()
    try:
        with patch.object(MujocoRenderer, 'render', side_effect=AssertionError('No GL in CPU source prerequisite')), \
                patch('mujoco.Renderer', side_effect=AssertionError('No GL in CPU source prerequisite')):
            for registration in IDS:
                task = registration.removeprefix('swm/Fetch').removesuffix('-v3').removesuffix('Dense').removesuffix('Dict')
                reward = 'dense' if 'Dense' in registration else 'sparse'
                refs = [gym.make(registration) for _ in range(2)]
                candidate, restored = FetchAdapter(2, task=task, reward_type=reward), FetchAdapter(2, task=task, reward_type=reward)
                try:
                    def compare(result, expected):
                        for name, index in [('reward', 1), ('terminated', 2), ('truncated', 3)]:
                            exact_array(result[name].cpu().numpy(), np.asarray([r[index] for r in expected]), name)
                        exact_array(result['success'].cpu().numpy(), np.asarray([r[4]['is_success'] for r in expected]), 'success')
                        full_compare(result['state'].cpu(), refs)

                    for seed in (17, 42):
                        for dtype in (np.float32, np.float64):
                            for lane, ref in enumerate(refs):
                                ref.reset(seed=seed + lane, options={'variation': []})
                            full_compare(candidate.reset(seed), refs)
                            restored.reset(seed + 1000)
                            actions = np.random.default_rng(90000 + seed).uniform(-1, 1, (16, 2, 4)).astype(dtype)
                            for action in actions[:8]:
                                expected = [ref.step(action[i]) for i, ref in enumerate(refs)]
                                compare(candidate.step(action), expected)
                            state = candidate.get_state()
                            held = state.clone()
                            for action in -actions[:5]:
                                restored.step(action)
                            restored.set_state(state)
                            equal_state(restored.get_state(), state)
                            # Ordinary stepping must never call set_state; this
                            # also prevents a hidden per-step oracle teleport.
                            with patch.object(candidate, 'set_state', side_effect=AssertionError('No implicit restore')), \
                                    patch.object(restored, 'set_state', side_effect=AssertionError('No implicit restore')):
                                for action in actions[8:]:
                                    expected = [ref.step(action[i]) for i, ref in enumerate(refs)]
                                    compare(candidate.step(action), expected)
                                    compare(restored.step(action), expected)
                            equal_state(state, held)
                            for ref in refs:
                                ref.reset(options={'variation': []})
                            full_compare(candidate.reset(), refs)
                            full_compare(restored.reset(), refs)
                            report['cases'].append(dict(registration=registration, seed=seed, action_dtype=np.dtype(dtype).name,
                                batch=2, prefix_steps=8, restored_continuation_steps=8, source_task_and_full_state='PASS',
                                different_history_restore='PASS', owned_state='PASS', unseeded_reset_rng='PASS',
                                ordinary_step_no_restore='PASS'))
                    save()
                    print(registration, 'four independent B2 source/restore cases PASS', flush=True)
                finally:
                    candidate.close()
                    restored.close()
                    for ref in refs:
                        ref.close()
        assert len(report['cases']) == 48
        report['status'] = 'PASS'
    except BaseException as error:
        report.update(status='FAIL', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()

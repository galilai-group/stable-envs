"""Oscar-only sampling diagnostic; no acceptance or throughput claims."""

import argparse
import contextlib
import hashlib
import json
import os
import platform
from pathlib import Path

import cv2
import numpy as np
import torch

import stable_envs as se
from scripts.benchmarks.online_stepping.classic_source_pipeline import image_errors
from scripts.benchmarks.online_stepping.dmc_source_pipeline import require_gl
from scripts.benchmarks.online_stepping.fetch_parallel_source import ParallelFetchSource, import_source
from scripts.benchmarks.online_stepping.fetch_source_pipeline import source_state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', '--domain', dest='task', choices=('Reach', 'Push', 'Slide'), required=True)
    p.add_argument('--device', '--devices', dest='device', choices=('cpu', 'cuda'), required=True)
    p.add_argument('--verify-gl', action='store_true')
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if platform.system() != 'Linux' or not os.environ.get('SLURM_JOB_ID') or not args.verify_gl:
        p.error('Requires an Oscar compute allocation and --verify-gl')
    manifest = json.loads(args.source_manifest.read_text())
    assert manifest['commit'] == '176d54532b3c0a251a4f509b74f2fdacc143a63d'
    for rel, digest in manifest['files'].items():
        assert hashlib.sha256((args.source_root / rel).read_bytes()).hexdigest() == digest, rel
    import_source(args.source_root)
    from scripts.recovery.fetch_public_gate import equal_state

    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    registration = f'swm/Fetch{args.task}-v3'
    report = dict(status='RUNNING', diagnostic_only=True, registration=registration,
                  device=args.device, host=platform.node(), job=os.environ['SLURM_JOB_ID'],
                  source_internal_size=480, output_size=64, cases=[], source_files=manifest['files'])
    report['probe_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    try:
        with ParallelFetchSource(registration, args.source_root, 4, 1, 1,
                                 frame_indices=(0,), render_size=64) as source, \
                contextlib.closing(se.make('Fetch' + args.task, num_envs=4, device=args.device,
                                          seed=17, size=64, backend='opengl')) as env:
            source.enable_render_audit()
            report['source_gl'] = list(source.renderer_info())
            for identity in report['source_gl']:
                require_gl(identity, args.device)
            equal_state(env.get_state().cpu(), source_state(source.snapshot()))
            actions = np.random.default_rng(31).uniform(-1, 1, (4, 1, 4))
            for step in range(2):
                expected = source.step(actions, args.device)['frames'].cpu()
                actual = env.step(actions[:, 0], states=None, frames=None)
                full = actual['state']
                equal_state(full.cpu(), source_state(source.snapshot()))
                images = {'source': expected[:, 0].numpy()}
                for label, size, ss in (('native64', 64, 1), ('ss2', 64, 2), ('ss4', 64, 4),
                                        ('native480_linear64', 480, 1)):
                    frames = env.render(full, size=size, supersample=ss).cpu()
                    if size == 480:
                        frames = torch.from_numpy(np.stack([cv2.resize(frame.numpy(), (64, 64))
                                                            for frame in frames]))
                    images[label] = frames.numpy()
                    report['cases'].append(dict(step=step + 1, method=label,
                                                 per_frame_mse=image_errors(frames[:, None], expected)))
                    equal_state(env.get_state(), full)
                np.savez_compressed(args.output / f'frames-{step + 1}.npz', **images)
        report['status'] = 'COMPLETE_DIAGNOSTIC_ONLY'
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

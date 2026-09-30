"""Alternating baseline/candidate/source experiment, same inputs and output device."""
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path


def load(name):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def probe(ours, source, env, batch, output, destination, sync, device):
    import torch
    scene = device == 'cuda'
    if scene:
        from stable_envs.envs.ogbench import manipulation_opengl as gl
        candidate = load('atlas')
        original = gl.render
        def select(mode):
            gl.render = original if mode in ('baseline', 'source') else candidate.render
            if mode.startswith('atlas'):
                candidate.TILE_CAP = int(mode[5:])
        variants = ['atlas16', 'atlas64']
    else:
        candidate = load('sparse')
        select = lambda mode: candidate.install(mode == 'sparse')
        variants = ['sparse']
    report = {'batch': batch, 'destination': destination, 'variants': {}, 'timings': {}}
    def stats(values):
        return dict(samples_ms=values, mean_ms=statistics.mean(values), sd_ms=statistics.stdev(values),
                    fps=batch * 1000 / statistics.mean(values))
    def quality(image, anchor):
        image, anchor = image.cpu(), anchor.cpu()
        mse = [float(((x.double() - y.double()) / 255).square().mean()) for x,y in zip(image,anchor)]
        return dict(mean=statistics.mean(mse), maximum=max(mse), exact=torch.equal(image,anchor))
    try:
        select('baseline')
        anchor = ours().clone(); sync()
        reference = source(); sync()
        report['source_mse'] = quality(anchor, reference)
        report['baseline_image_sha256'] = hashlib.sha256(anchor.cpu().numpy().tobytes()).hexdigest()
        for mode in variants:
            select(mode)
            image = ours(); sync()
            report['variants'][mode] = quality(image, anchor)
            # Preserve evidence if atlas coordinate rounding changes pixels.
            report['variants'][mode]['source_mse'] = quality(image, reference)
            assert report['variants'][mode]['maximum'] < 1e-5, report
            if mode == 'sparse':
                assert torch.equal(image, anchor), report
        samples = {mode: [] for mode in ['baseline', *variants, 'source']}
        for repeat in range(5):
            modes = list(samples)
            if repeat % 2:
                modes.reverse()
            for mode in modes:
                select(mode)
                sync(); start=time.perf_counter()
                image = source() if mode == 'source' else ours()
                sync(); samples[mode].append((time.perf_counter()-start)*1000)
                if mode != 'source':
                    assert quality(image, anchor)['maximum'] < 1e-5
        report['timings'] = {mode:stats(values) for mode,values in samples.items()}
        report['status'] = 'PASS'
        return {'ours':report['timings']['baseline'], 'source':report['timings']['source']}
    except Exception as error:
        report['status']='ERROR'; report['error']=str(error)
        raise
    finally:
        select('baseline')
        (output/f'structural_b{batch}_{destination}.json').write_text(json.dumps(report,indent=2)+'\n')

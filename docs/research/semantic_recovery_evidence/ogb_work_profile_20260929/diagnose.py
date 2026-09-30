"""Warm exact-workload diagnosis; no production edits or acceptance claims."""
import cProfile
import functools
import json
import pstats
import time
from collections import Counter


def diagnose(ours, source, value, output):
    import mujoco
    import torch
    import warp as wp
    from stable_envs.envs.ogbench import manipulation as frozen
    from stable_envs.envs.ogbench import manipulation_renderer as wired
    from stable_envs.envs.ogbench import manipulation_opengl as gl
    from stable_envs._adapters import ogbench_render_state as pixels

    result = {'note': 'Exact paired saved-State batch256; timings exclude loading/projection. Instrumented wall times include profiler overhead; nested cumulative times must not be summed.'}
    torch.cuda.synchronize()
    anchor = ours().clone()
    torch.cuda.synchronize()
    originals = []
    entries, stack, contexts = {}, [], Counter()

    def wrap(obj, name, label):
        old = getattr(obj, name)
        @functools.wraps(old)
        def call(*args, **kwargs):
            tag = label
            if label == 'context':
                key = (args[0], int(args[1]), int(args[2]))
                tag += '.hit' if key in frozen._CTX_CACHE else '.miss'
                contexts[str(key)] += 1
            node = [time.perf_counter_ns(), 0]
            stack.append(node)
            try:
                return old(*args, **kwargs)
            finally:
                elapsed = time.perf_counter_ns() - node[0]
                stack.pop()
                if stack:
                    stack[-1][1] += elapsed
                row = entries.setdefault(tag, {'calls': 0, 'inclusive_ms': 0, 'exclusive_ms': 0})
                row['calls'] += 1
                row['inclusive_ms'] += elapsed / 1e6
                row['exclusive_ms'] += (elapsed - node[1]) / 1e6
        originals.append((obj, name, old))
        setattr(obj, name, call)

    for obj, name, tag in [
        (wired, 'render', 'renderer_boundary'),
        (wired, 'configuration_key', 'configuration_hash'),
        (wired, 'configured_bundle', 'configured_bundle'),
        (pixels, 'validate', 'validate_render_state'),
        (wired._ConfiguredRenderer, 'draw', 'pass.draw'),
        (frozen.OGBManipBase, '_render', 'pass.render'),
        (frozen, '_context', 'context'),
        (frozen.MW, 'create_render_context', 'build_render_context'),
        (frozen, '_frame', 'context.warmup_capture_frame'),
        (frozen.OGBManipBase, '_bake_plate', 'bake_static_plate'),
        (wp, 'capture_launch', 'warp_graph_launch'),
        (wp, 'synchronize', 'warp_wait'),
        (torch.cuda, 'synchronize', 'torch_wait'),
        (gl, '_renderer', 'opengl_lookup'),
        (mujoco, 'mj_forward', 'mj_forward'),
        (mujoco.Renderer, 'update_scene', 'opengl_update_scene'),
        (mujoco.Renderer, 'render', 'opengl_draw_readback'),
    ]:
        wrap(obj, name, tag)
    before = {k: getattr(frozen._CTX_CACHE, k) for k in ('hits', 'misses', 'evictions', 'capacity')}
    try:
        samples = []
        for _ in range(3):
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            actual = ours()
            torch.cuda.synchronize()
            samples.append((time.perf_counter_ns() - start) / 1e6)
            assert torch.equal(actual, anchor)
        result.update(instrumented_ms=samples, sections=entries, context_calls=dict(contexts))
        result['cache_before'] = before
        result['cache_after'] = {k: getattr(frozen._CTX_CACHE, k) for k in before}
    finally:
        for obj, name, old in reversed(originals):
            setattr(obj, name, old)
    for label, fn in [('ours', ours), ('source', source)]:
        pr = cProfile.Profile()
        torch.cuda.synchronize()
        pr.enable()
        fn()
        torch.cuda.synchronize()
        pr.disable()
        pr.dump_stats(str(output / (label + '.prof')))
        stats = pstats.Stats(pr)
        result[label + '_functions'] = [dict(file=k[0], line=k[1], function=k[2], primitive_calls=v[0], calls=v[1], self_ms=v[2]*1000, cumulative_ms=v[3]*1000) for k,v in stats.stats.items()]
    result['cuda_memory_allocated_bytes'] = torch.cuda.memory_allocated()
    result['cuda_memory_reserved_bytes'] = torch.cuda.memory_reserved()
    (output / 'diagnosis.json').write_text(json.dumps(result, indent=2) + '\n')

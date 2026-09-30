"""Residual Scene CPU-side call attribution; GPU sensors are collected separately."""
import cProfile
import functools
import json
import pstats
import time


def diagnose(ours, source, output, batch):
    import mujoco
    import torch
    from stable_envs.envs.ogbench import manipulation_opengl as gl
    from stable_envs._adapters import ogbench_render_state as pixels

    output.mkdir(exist_ok=False)
    report = {'batch': batch, 'note': 'Diagnostic wall times include wrappers. Draw/readback wall time includes driver work and GPU waits, not GPU kernel occupancy. Nested inclusive times must not be added.'}
    for label, fn in [('ours', ours), ('source', source)]:
        torch.cuda.synchronize()
        anchor = fn()[:16].detach().cpu().clone()
        entries, stack, originals = {}, [], []
        def wrap(obj, name, tag):
            old = getattr(obj, name)
            @functools.wraps(old)
            def call(*args, **kwargs):
                start = time.perf_counter_ns(); stack.append([start, 0])
                try: return old(*args, **kwargs)
                finally:
                    node = stack.pop(); elapsed = time.perf_counter_ns()-node[0]
                    if stack: stack[-1][1] += elapsed
                    row = entries.setdefault(tag, {'calls':0,'inclusive_ms':0.,'exclusive_ms':0.})
                    row['calls'] += 1; row['inclusive_ms'] += elapsed/1e6; row['exclusive_ms'] += (elapsed-node[1])/1e6
            originals.append((obj,name,old)); setattr(obj,name,call)
        for obj,name,tag in [(gl,'render','render_boundary'),(pixels,'validate','validate'),(gl,'_configuration_groups','group_configuration'),(gl,'_renderer','renderer_lookup'),(gl,'_apply_render_button_colors','button_colors'),(mujoco,'mj_kinematics','kinematics'),(mujoco,'mj_camlight','camera_light'),(mujoco,'mj_forward','full_forward'),(mujoco.Renderer,'update_scene','update_scene'),(mujoco.Renderer,'render','draw_readback'),(mujoco,'mjr_render','draw_submit'),(mujoco,'mjr_readPixels','readback'),(torch.cuda,'synchronize','torch_wait')]:
            wrap(obj,name,tag)
        samples=[]
        try:
            for _ in range(3):
                torch.cuda.synchronize();start=time.perf_counter_ns();cstart=time.process_time_ns()
                image=fn();torch.cuda.synchronize()
                samples.append({'wall_ms':(time.perf_counter_ns()-start)/1e6,'cpu_ms':(time.process_time_ns()-cstart)/1e6})
                assert int((image[:16].detach().cpu().to(torch.int16)-anchor.to(torch.int16)).abs().max())<=1
                del image
        finally:
            for obj,name,old in reversed(originals):setattr(obj,name,old)
        pr=cProfile.Profile();torch.cuda.synchronize();pr.enable();fn();torch.cuda.synchronize();pr.disable()
        pr.dump_stats(str(output/f'{label}.prof'))
        functions=[dict(file=k[0],line=k[1],function=k[2],calls=v[1],self_ms=v[2]*1000,cumulative_ms=v[3]*1000) for k,v in pstats.Stats(pr).stats.items()]
        report[label]={'samples':samples,'sections':entries,'functions':functions}
    (output/'diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')

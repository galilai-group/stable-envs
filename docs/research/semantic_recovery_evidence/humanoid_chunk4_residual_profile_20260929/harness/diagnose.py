"""Warmed Humanoid CPU residual profile; no production changes or GPU work."""
import cProfile
import json
import math
import os
import pstats
import resource
import time
from collections import defaultdict
from contextlib import ExitStack
from unittest.mock import patch


def memory():
    status = {}
    for line in open('/proc/self/status'):
        if line.startswith(('VmRSS:', 'VmHWM:', 'Threads:')):
            k, v = line.split(':', 1)
            status[k] = v.strip()
    return status


def diagnose(ours, source, output, batch):
    import torch
    from stable_envs.render import cpu3d as c3, raycast3d as r3
    from stable_envs.envs.dmcontrol.trackcom import DMControlTrackCom
    from stable_envs.envs.dmcontrol.humanoid import DMCHumanoidWalk

    output.mkdir(exist_ok=False)
    assert DMCHumanoidWalk.CPU_RENDER_CHUNK == 4
    result = dict(batch=batch, chunk=4, torch_threads=torch.get_num_threads(), affinity=sorted(os.sched_getaffinity(0)),
                  note='Warmed saved-State CPU render. No rollouts/disk reads/State projection in timing. Source installation excluded. Instrumentation timings are diagnostic; unprofiled paired timings authoritative.',
                  memory_before=memory())
    def save():
        (output/'diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
    for label, fn in [('ours',ours),('source',source)]:
        anchor=fn().clone()
        samples=[]
        for _ in range(3):
            before=resource.getrusage(resource.RUSAGE_SELF)
            start=time.perf_counter_ns()
            actual=fn()
            elapsed=(time.perf_counter_ns()-start)/1e6
            after=resource.getrusage(resource.RUSAGE_SELF)
            cpu_ms=((after.ru_utime-before.ru_utime)+(after.ru_stime-before.ru_stime))*1000
            samples.append(dict(wall_ms=elapsed,cpu_ms=cpu_ms,cpu_one_core_percent=100*cpu_ms/elapsed,
                                cpu_allocation_percent=100*cpu_ms/elapsed/len(os.sched_getaffinity(0)),
                                minor_faults=after.ru_minflt-before.ru_minflt,major_faults=after.ru_majflt-before.ru_majflt,
                                voluntary_context_switches=after.ru_nvcsw-before.ru_nvcsw,
                                involuntary_context_switches=after.ru_nivcsw-before.ru_nivcsw))
            assert torch.equal(anchor,actual),label+' repeat'
        result[label+'_unprofiled']=samples
        result[label+'_memory']=memory()
        prof=cProfile.Profile();start=time.perf_counter_ns();prof.enable();actual=fn();prof.disable()
        result[label+'_cprofile_ms']=(time.perf_counter_ns()-start)/1e6
        assert torch.equal(anchor,actual)
        prof.dump_stats(str(output/(label+'.prof')))
        result[label+'_functions']=[dict(file=k[0],line=k[1],function=k[2],calls=v[1],self_ms=v[2]*1000,cumulative_ms=v[3]*1000)
                                   for k,v in pstats.Stats(prof).stats.items()]
        save()

    totals=defaultdict(lambda:dict(calls=0,inclusive_ms=0.,exclusive_ms=0.))
    stack=[]; counts=defaultdict(int)
    def wrap(name,fn):
        def instrumented(*args,**kwargs):
            row=[time.perf_counter_ns(),0]
            stack.append(row)
            if name=='bounds': counts['bound_ray_primitive_candidates']+=math.prod(args[3].shape[:-1])
            if name=='intersect': counts['evaluated_ray_primitive_candidates']+=math.prod(args[4].shape[:-1])
            try:return fn(*args,**kwargs)
            finally:
                elapsed=time.perf_counter_ns()-row[0];stack.pop()
                if stack:stack[-1][1]+=elapsed
                totals[name]['calls']+=1;totals[name]['inclusive_ms']+=elapsed/1e6
                totals[name]['exclusive_ms']+=(elapsed-row[1])/1e6
        return instrumented
    with ExitStack() as patches:
        for obj,attr,name in [
            (DMControlTrackCom,'_render_cpu','wrapper'),(DMControlTrackCom,'_cpu_poses','pose_preparation'),
            (DMControlTrackCom,'_cpu_scene','scene_lookup'),(c3,'render','cpu_render'),(c3,'_render_chunk','chunk'),
            (c3,'_mover_window','primary_window'),(c3,'_light_footprint','shadow_footprint'),
            (c3,'_any_blocking_flat','shadow_flatten'),(c3,'bilinear','texture_bilinear'),(c3,'_normalize','normalize'),
            (r3,'cast_nearest_torch','primary_cast'),(r3,'any_blocking_torch','shadow_cast'),
            (r3,'_rays_near','bounds'),(r3,'intersect','intersect'),
            (r3,'_capsule','capsule'),(r3,'_sphere','sphere'),(r3,'_ellipsoid','ellipsoid'),
            (r3,'_cylinder','cylinder'),(r3,'_box','box')]:
            patches.enter_context(patch.object(obj,attr,wrap(name,getattr(obj,attr))))
        start=time.perf_counter_ns();actual=ours();result['section_elapsed_ms']=(time.perf_counter_ns()-start)/1e6
    assert torch.equal(actual,ours())
    result['sections']=dict(totals);result['ray_counts']=dict(counts);save()
    if batch==64:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True,profile_memory=True) as prof:
            ours()
        result['torch_operators']=sorted([dict(name=e.key,calls=e.count,self_cpu_ms=e.self_cpu_time_total/1000,
                                               cpu_ms=e.cpu_time_total/1000,self_cpu_memory_bytes=e.self_cpu_memory_usage,
                                               cpu_memory_bytes=e.cpu_memory_usage)
                                         for e in prof.key_averages()],key=lambda e:e['self_cpu_ms'],reverse=True)
        result['torch_memory_note']='Operator memory fields count allocation/deallocation activity, not measured DRAM bandwidth or maximum live working set.'
    result['memory_after']=memory();result['status']='COMPLETE';save()

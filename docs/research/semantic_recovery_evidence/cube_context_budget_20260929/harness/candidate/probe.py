import hashlib
import json
import os
import subprocess
import time


def probe(ours, batch, output):
    import torch
    from stable_envs.envs.ogbench import manipulation as frozen
    cache = frozen._CTX_CACHE
    def stats():
        if hasattr(cache, 'stats'):
            return cache.stats()
        return dict(hits=cache.hits, misses=cache.misses, evictions=cache.evictions, entries=len(cache))
    before = stats()
    hashes, times = [], []
    for _ in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        image = ours()
        torch.cuda.synchronize()
        times.append((time.perf_counter_ns()-start)/1e6)
        hashes.append(hashlib.sha256(image.cpu().numpy().tobytes()).hexdigest())
    after = stats()
    result=dict(batch=batch, process=os.getpid(), before=before, after=after, samples_ms=times, image_sha256=hashes,
      nvidia_memory=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout)
    (output/f'reuse_b{batch}.json').write_text(json.dumps(result,indent=2)+'\n')
    assert len(set(hashes)) == 1, 'Repeated Cube rendering changed pixels'
    if hasattr(cache,'stats'):
        assert after['misses'] == before['misses'], 'Warm workload rebuilt GPU contexts'
        assert after['evictions'] == before['evictions'], 'Warm workload evicted GPU contexts'
        assert all(v <= after['budget_bytes'][k] for k,v in after['retained_bytes'].items())

"""Device telemetry for separate sustained render phases, not benchmark FPS.

Call only inside an allocated Oscar GPU process after the renderers are warm::

    ours = run_phase('ours_b256_cpu', ours_fn, torch.cuda.synchronize, output_dir)
    source = run_phase('source_b256_cpu', source_fn, torch.cuda.synchronize, output_dir)

Creates output_dir/label without overwriting. Imports Torch only to resolve a
missing UUID inside run_phase. Standard NVML/nvidia-smi utilization is sampled
busy time, not SM occupancy, peak-bandwidth utilization, FLOPs or MFU.

Definitions: https://docs.nvidia.com/deploy/nvidia-smi/index.html#utilization
Detailed SM/graphics counters require a supported, permitted Nsight metrics run:
https://docs.nvidia.com/nsight-systems/UserGuide/index.html#gpu-metrics
"""
from __future__ import annotations

import csv
import datetime as dt
import glob
import json
import math
import os
import shutil
import socket
import statistics
import subprocess
import threading
import time
import uuid as uuid_module
import xml.etree.ElementTree as ET
from pathlib import Path

FIELDS = (
    'timestamp', 'uuid', 'pci.bus_id', 'name', 'utilization.gpu',
    'utilization.memory', 'memory.used', 'memory.total', 'power.draw',
    'power.limit', 'temperature.gpu', 'clocks.current.graphics',
    'clocks.current.sm', 'clocks.current.memory', 'pstate',
)
NUMERIC = {
    'utilization.gpu': 'gpu_busy_percent',
    'utilization.memory': 'memory_busy_percent',
    'memory.used': 'vram_used_mib', 'memory.total': 'vram_total_mib',
    'power.draw': 'power_watts', 'power.limit': 'power_limit_watts',
    'temperature.gpu': 'temperature_c', 'clocks.current.graphics': 'graphics_clock_mhz',
    'clocks.current.sm': 'sm_clock_mhz', 'clocks.current.memory': 'memory_clock_mhz',
}


def _utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _stats(values):
    if not values:
        return {'count': 0, 'mean': None, 'sd': None, 'p95': None, 'min': None, 'max': None}
    ordered = sorted(values)
    at = (len(ordered) - 1) * 0.95
    low, high = math.floor(at), math.ceil(at)
    p95 = ordered[low] + (ordered[high] - ordered[low]) * (at - low)
    return {'count': len(values), 'mean': statistics.mean(values),
            'sd': statistics.stdev(values) if len(values) > 1 else None,
            'p95': p95, 'min': min(values), 'max': max(values)}


def _command(args, timeout=15):
    try:
        done = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {'command': args, 'returncode': done.returncode,
                'stdout': done.stdout, 'stderr': done.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'command': args, 'returncode': None, 'error': repr(error)}


def _resolve_uuid(value):
    if value is None:
        import torch
        value = getattr(torch.cuda.get_device_properties(torch.cuda.current_device()), 'uuid', None)
        if value is None:
            raise RuntimeError('CUDA device UUID unavailable; pass device_uuid explicitly. Physical GPU index guessing is prohibited.')
    if isinstance(value, bytes):
        value = str(uuid_module.UUID(bytes=value)) if len(value) == 16 else value.decode()
    value = str(value)
    if not value.startswith('GPU-'):
        value = 'GPU-' + str(uuid_module.UUID(value))
    return value


def _process_snapshot(executable, gpu_uuid):
    record = _command([executable, '--id=' + gpu_uuid, '-q', '-x'])
    record.update(utc=_utc(), monotonic_ns=time.monotonic_ns(), processes=[], gpu_uuid=gpu_uuid)
    if record['returncode'] != 0:
        return record
    try:
        doc = ET.fromstring(record.pop('stdout'))
        devices = doc.findall('gpu')
        if len(devices) != 1 or devices[0].findtext('uuid') != gpu_uuid:
            raise ValueError('nvidia-smi process snapshot did not resolve exactly the requested UUID')
        gpu = devices[0]
        record['name'] = gpu.findtext('product_name')
        record['driver_version'] = doc.findtext('driver_version')
        record['pci_bus_id'] = gpu.findtext('pci/pci_bus_id')
        for proc in gpu.findall('processes/process_info'):
            pid = proc.findtext('pid')
            record['processes'].append({'pid': int(pid) if pid and pid.isdigit() else pid,
                                       'type': proc.findtext('type'),
                                       'name': proc.findtext('process_name'),
                                       'used_memory': proc.findtext('used_memory')})
        node = gpu.find('processes')
        record['process_visibility'] = 'reported' if node is not None and (node.text or '').strip() not in {'N/A', 'Not Supported'} else 'unavailable'
    except (ValueError, ET.ParseError) as error:
        record['parse_error'] = repr(error)
    return record


def probe_tools(output_dir):
    """Read-only capability inventory; call inside the allocated GPU job."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    result = {'utc': _utc(), 'host': socket.gethostname(), 'pid': os.getpid(),
              'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
              'nvidia_smi': shutil.which('nvidia-smi'), 'nsys': shutil.which('nsys')}
    if not result['nsys']:
        matches = glob.glob('/usr/local/cuda*/bin/nsys') + glob.glob('/usr/local/cuda*/nsight-systems*/bin/nsys')
        result['nsys'] = next((p for p in matches if os.access(p, os.X_OK)), None)
    params = Path('/proc/driver/nvidia/params')
    try:
        result['driver_profiling_parameters'] = [line for line in params.read_text().splitlines() if 'Profil' in line]
    except OSError as error:
        result['driver_profiling_parameters_error'] = repr(error)
    if result['nvidia_smi']:
        result['nvidia_smi_version'] = _command([result['nvidia_smi'], '--version'])
        result['dmon_help'] = _command([result['nvidia_smi'], 'dmon', '--help'])
        result['pmon_help'] = _command([result['nvidia_smi'], 'pmon', '--help'])
    if result['nsys']:
        result['nsys_version'] = _command([result['nsys'], '--version'])
        result['nsys_environment'] = _command([result['nsys'], 'status', '--environment'])
        result['nsys_metrics_inventory'] = _command([result['nsys'], 'profile', '--gpu-metrics-devices=help'])
    result['interpretation'] = 'No metrics profiler was launched. A listed GPU/metric set does not prove permission to collect performance counters. No elevated privileges requested.'
    (directory / 'telemetry_tools.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def run_phase(label, render_fn, sync_fn, output_dir, *, device_uuid=None, duration_s=12, interval_ms=200):
    """Sample one physical GPU through a sustained, synchronized render phase.

    Diagnostic timings include telemetry overhead and are not headline FPS.
    Samples in the first/last 1 second are excluded from the metric summary.
    nvidia-smi's sensor window is product-dependent (roughly 1/6 to 1 second),
    so polling at 200 ms does not create independent 200 ms hardware samples.
    """
    if duration_s < 3 or not 100 <= interval_ms <= 1000:
        raise ValueError('Use duration_s >= 3 and interval_ms in [100, 1000]')
    if not label or Path(label).name != label:
        raise ValueError('label must be one nonempty path component')
    executable = shutil.which('nvidia-smi')
    if executable is None:
        raise RuntimeError('nvidia-smi is unavailable in this allocated job')
    gpu_uuid = _resolve_uuid(device_uuid)
    directory = Path(output_dir) / label
    directory.mkdir(parents=True, exist_ok=False)
    command = [executable, '--id=' + gpu_uuid, '--query-gpu=' + ','.join(FIELDS),
               '--format=csv,noheader,nounits', '--loop-ms=' + str(interval_ms)]
    warm = render_fn()
    sync_fn()
    del warm
    snapshots = [_process_snapshot(executable, gpu_uuid)]
    first = snapshots[0]
    if first['returncode'] != 0 or first.get('parse_error'):
        (directory / 'identity_error.json').write_text(json.dumps(first, indent=2) + '\n')
        raise RuntimeError('Could not verify telemetry GPU UUID; see identity_error.json')
    samples, parse_errors = [], []
    stop, ready = threading.Event(), threading.Event()
    errors_file = (directory / 'sampler.stderr').open('w')
    sampler = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors_file, text=True, bufsize=1)

    def read_samples():
        with (directory / 'samples.jsonl').open('w') as stream:
            for line in sampler.stdout:
                record = {'received_utc': _utc(), 'received_monotonic_ns': time.monotonic_ns(), 'raw': line.rstrip()}
                values = next(csv.reader([line]), [])
                if len(values) != len(FIELDS):
                    record['parse_error'] = 'Unexpected column count'
                    parse_errors.append(record)
                else:
                    raw = dict(zip(FIELDS, [v.strip() for v in values]))
                    record['raw_fields'] = raw
                    if raw['uuid'] != gpu_uuid:
                        record['parse_error'] = 'Unexpected GPU UUID'
                        parse_errors.append(record)
                    else:
                        for field, key in NUMERIC.items():
                            try:
                                number = float(raw[field])
                                record[key] = number if math.isfinite(number) else None
                            except ValueError:
                                record[key] = None
                        samples.append(record)
                stream.write(json.dumps(record) + '\n')
                stream.flush()
                ready.set()

    def read_processes():
        while not stop.wait(2):
            snapshots.append(_process_snapshot(executable, gpu_uuid))

    reader = threading.Thread(target=read_samples, daemon=True)
    process_reader = threading.Thread(target=read_processes, daemon=True)
    reader.start()
    ready.wait(timeout=3)
    process_reader.start()
    sync_fn()
    started = time.monotonic_ns()
    started_utc = _utc()
    cpu_started = time.process_time()
    calls = 0
    failure = None
    try:
        while (time.monotonic_ns() - started) / 1e9 < duration_s:
            output = render_fn()
            sync_fn()
            del output
            calls += 1
    except BaseException as error:
        failure = repr(error)
        raise
    finally:
        ended = time.monotonic_ns()
        ended_utc = _utc()
        cpu_seconds = time.process_time() - cpu_started
        stop.set()
        sampler.terminate()
        try:
            sampler.wait(timeout=5)
        except subprocess.TimeoutExpired:
            sampler.kill()
            sampler.wait()
        reader.join(timeout=5)
        process_reader.join(timeout=20)
        errors_file.close()
        snapshots.append(_process_snapshot(executable, gpu_uuid))
        retained = [s for s in samples if started + 1_000_000_000 <= s['received_monotonic_ns'] <= ended - 1_000_000_000]
        other_pids = sorted({p['pid'] for snapshot in snapshots for p in snapshot['processes']
                             if isinstance(p.get('pid'), int) and p['pid'] != os.getpid()})
        wall_seconds = (ended - started) / 1e9
        summary = {
            'label': label, 'status': 'ERROR' if failure else ('COMPLETE' if any(s.get('gpu_busy_percent') is not None for s in retained) else 'NO_VALID_GPU_UTILIZATION'),
            'error': failure, 'host': socket.gethostname(), 'pid': os.getpid(), 'gpu_uuid': gpu_uuid,
            'gpu_name': first.get('name'), 'pci_bus_id': first.get('pci_bus_id'),
            'started_utc': started_utc, 'ended_utc': ended_utc,
            'started_monotonic_ns': started, 'ended_monotonic_ns': ended,
            'phase_seconds': wall_seconds, 'render_calls': calls, 'process_cpu_seconds': cpu_seconds,
            'process_cpu_duty_percent': cpu_seconds / wall_seconds * 100,
            'poll_interval_ms': interval_ms, 'edge_exclusion_seconds': 1,
            'sample_count_total': len(samples), 'sample_count_retained': len(retained),
            'metrics': {key: _stats([s[key] for s in retained if s.get(key) is not None]) for key in NUMERIC.values()},
            'possible_cross_process_contamination': bool(other_pids), 'other_gpu_pids': other_pids,
            'process_snapshot_count': len(snapshots),
            'process_visibility_complete': all(s.get('process_visibility') == 'reported' and not s.get('parse_error') for s in snapshots),
            'process_attribution_limit': 'Device-wide counters. Process lists sampled every 2 s can miss short-lived competitors; absence of another PID is not proof of exclusive execution.',
            'sampler_command': command, 'sampler_returncode': sampler.returncode,
            'parse_errors': parse_errors,
            'interpretation': 'GPU busy% is time activity; memory busy% is time device memory was read/written, not bandwidth saturation. Neither measures SM occupancy, FLOPs, MFU, or separate graphics/compute/copy-engine saturation. Polls can repeat the same hardware sampling window. Phase timings include telemetry overhead; use independent uninstrumented paired timings for FPS.',
            'sources': ['https://docs.nvidia.com/deploy/nvidia-smi/index.html#utilization',
                        'https://docs.nvidia.com/nsight-systems/UserGuide/index.html#gpu-metrics'],
        }
        (directory / 'processes.json').write_text(json.dumps(snapshots, indent=2) + '\n')
        (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary

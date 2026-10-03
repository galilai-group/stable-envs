"""Reconcile online source performance evidence without substituting internal controls.

This checkpoint audits the completed five-registration classic campaign. It
revalidates raw reports against their frozen candidate and terminal accounting;
other registrations remain explicit gaps. It does not rerun any benchmark or
change acceptance. Output directories must be new.
"""

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

EVIDENCE = Path('docs/research/semantic_recovery_evidence')
CLASSIC = EVIDENCE / 'classic_parallel_source_20261003/completed'
REGISTRATIONS = {
    'cartpole': 'swm/CartPoleControl-v1',
    'pendulum': 'swm/PendulumControl-v1',
    'acrobot': 'swm/AcrobotControl-v1',
    'mountain_car': 'swm/MountainCarControl-v0',
    'mountain_car_continuous': 'swm/MountainCarContinuousControl-v0',
}


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def terminal_jobs(text):
    jobs = {}
    for line in text.splitlines():
        task, job, status, exitcode, elapsed, host = line.split('|')
        if '.' in task:
            continue
        assert status == 'COMPLETED' and exitcode == '0:0', line
        assert job not in jobs
        jobs[job] = host
    return jobs


def comparison_kind(row):
    assert row['source_gpu_supported'] == 'False'
    assert row['source_render_device'] == 'cpu'
    assert row['output_device'] in ('cpu', 'cuda')
    return 'CPU_VS_CPU' if row['output_device'] == 'cpu' else 'GPU_VS_CPU_WITH_CUDA_DELIVERY_SOURCE_GPU_UNSUPPORTED'


def gap(row):
    family, name = row['family'], row['historical_id']
    if family == 'craftax':
        return 'DEFERRED', 'Craftax optimization excluded from this goal.'
    if row['state_status'] != 'VERIFIED':
        return 'CORRECTNESS_BLOCKED', 'Independent semantic gate required before optimization.'
    if family == 'classic_control':
        return 'MEASURED_QUALIFIED', 'One parent Torch thread; eight-thread source rerun 6971892 remains uncollected. B64/256, T1/8; not broad-state or sustained acceptance.'
    if family == 'fetch':
        if 'PickAndPlace' in name:
            return 'SOURCE_CAMPAIGN_PREPARED_NO_FRAMES', 'B64/T1/T8 no-frame P1/P8/public comparison prepared, not submitted. Source rendering changes continuation; rendered comparison remains unresolved.'
        return 'SOURCE_CAMPAIGN_PREPARED', 'B64/T1/T8 P1/P8/public source comparison prepared, not submitted; registration-specific CPU/CUDA image gates must pass before timing. Existing performance sweeps use library controls.'
    if family == 'dmcontrol':
        if 'Quadruped' in name:
            return 'SOURCE_COMPARISON_MISSING', 'Renderer-history discrepancy unresolved; local route retained. No completed online source comparison.'
        if any(task in name for task in ('Acrobot', 'Cartpole', 'Pendulum')):
            return 'SOURCE_CAMPAIGN_PREPARED', 'Bounded B64 source comparison bundle prepared, not submitted; completed pipeline sweep uses a library control.'
        return 'SOURCE_COMPARISON_MISSING', 'CPU source transport prerequisite and library-control pipeline sweep complete; direct source rendering/timing remains open.'
    if family == 'ogbench':
        return 'SOURCE_IMAGE_GATE_PREPARED', 'Source CPU P1/P2 transport/terminal prerequisites complete; isolated source GL image gate prepared, not submitted. Actual GL nonmutation, source timing and resource lifetime remain open.'
    if family == 'navigation':
        return 'SOURCE_CAMPAIGN_PREPARED', 'RTX 3090 source comparison prepared, not submitted; local CPU smoke checks are not cluster performance results.'
    raise ValueError(f'Unclassified family: {family}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen-classic-code', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    matrix_path = Path('docs/research/semantic_recovery_matrix.json')
    routes_path = EVIDENCE / 'online_api_docs_20261003/noncraftax-routes.csv'
    matrix = json.loads(matrix_path.read_text())['rows']
    assert len(matrix) == len({r['historical_id'] for r in matrix}) == 42
    assert Counter(r['state_status'] for r in matrix) == {'VERIFIED': 39, 'INCOMPLETE': 3}
    routes = {r['registration']: r for r in read_csv(routes_path)}
    assert set(routes) == {r['historical_id'] for r in matrix if r['family'] != 'craftax'}
    jobs = terminal_jobs((CLASSIC / 'terminal-family.txt').read_text())
    assert len(jobs) == 5
    inputs = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (matrix_path, routes_path, CLASSIC / 'terminal-family.txt')}
    route_review_path = EVIDENCE / 'online_api_docs_20261003/review.json'
    route_review = json.loads(route_review_path.read_text())
    inputs[str(route_review_path)] = hashlib.sha256(route_review_path.read_bytes()).hexdigest()
    for rel, digest in route_review['files'].items():
        if rel.startswith('stable_envs/'):
            assert hashlib.sha256(Path(rel).read_bytes()).hexdigest() == digest, rel
            inputs[rel] = digest
    # Prepared evidence is not measured performance. Verify the archived inputs
    # before assigning a prepared status; never fill numeric cells from them.
    for directory in ('fetch_source_pipeline_20261003', 'dmc_source_pipeline_20261003',
                      'navigation_source_20261003', 'ogbench_source_isolation_20261003'):
        base = EVIDENCE / directory
        sums_file = base / 'SHA256SUMS.json'
        inputs[str(sums_file)] = hashlib.sha256(sums_file.read_bytes()).hexdigest()
        for rel, digest in json.loads(sums_file.read_text()).items():
            path = base / rel
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path
            inputs[str(path)] = digest
        deployment = json.loads((base / 'deployment.json').read_text())
        assert deployment['status'] in ('PREPARED_NOT_UPLOADED', 'NOT_UPLOADED_NOT_SUBMITTED')
    sums_path = CLASSIC / 'family-closure/SHA256SUMS.json'
    sums = json.loads(sums_path.read_text())
    for rel, digest in sums.items():
        path = CLASSIC / rel
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path
        inputs[str(path)] = digest
    measured = []
    with tempfile.TemporaryDirectory(prefix='online-source-audit-') as temporary:
        for domain, registration in REGISTRATIONS.items():
            report_path = CLASSIC / (domain + '.json')
            report = json.loads(report_path.read_text())
            assert report['domain'] == domain
            assert jobs[str(report['job'])] == report['host'].split('.')[0]
            output = Path(temporary) / (domain + '.csv')
            subprocess.run([sys.executable, 'scripts/benchmarks/online_stepping/summarize_classic_parallel_source.py',
                            str(report_path), str(output), '--source', str(args.frozen_classic_code)], check=True)
            assert output.read_bytes() == (CLASSIC / (domain + '.csv')).read_bytes()
            matched = all(hashlib.sha256(Path(rel).read_bytes()).hexdigest() == digest
                          for rel, digest in report['candidate_files'].items() if rel.startswith('stable_envs/'))
            for original in read_csv(output):
                row = dict(registration=registration, comparison_kind=comparison_kind(original),
                    measurement_scope='ONLINE_STEP_AND_SELECTED_RENDER',
                    ours_state_history=False, ours_required_final_state=True,
                    parent_torch_threads=1, measured_production_file_hashes_match=matched,
                    report=str(report_path), **original)
                # Empty FPS/MSE is missing by definition when no frames are
                # requested; never turn those cells into zero-error/FPS wins.
                if row['selection'] == 'none':
                    assert row['mse_mean'] == row['ours_fps_mean'] == row['source_serial_fps_mean'] == ''
                else:
                    assert all(math.isfinite(float(row[k])) and float(row[k]) > 0
                               for k in ('source_serial_fps_mean', 'source_parallel_fps_mean', 'ours_fps_mean'))
                measured.append(row)
    assert len(measured) == 120
    overview = []
    for entry in matrix:
        registration = entry['historical_id']
        status, remaining = gap(entry)
        row = dict(registration=registration, family=entry['family'], semantic_status=entry['state_status'],
            route=routes[registration]['route'] if registration in routes else 'deferred',
            online_source_status=status, batch='', horizon='', selection='',
            source_native_gpu_supported='', cpu_mse_mean='', cpu_mse_sd='', gpu_mse_mean='', gpu_mse_sd='',
            source_cpu_serial_fps_mean='', source_cpu_serial_fps_sd='',
            source_cpu_parallel_fps_mean='', source_cpu_parallel_fps_sd='',
            ours_cpu_fps_mean='', ours_cpu_fps_sd='', ours_cpu_vs_best_source='',
            source_gpu_fps_mean='', source_gpu_fps_sd='',
            source_cpu_cuda_delivery_parallel_fps_mean='', source_cpu_cuda_delivery_parallel_fps_sd='',
            ours_gpu_fps_mean='', ours_gpu_fps_sd='', ours_gpu_vs_best_cpu_cuda_delivery='',
            parent_torch_threads='', measured_production_file_hashes_match='', evidence='', remaining=remaining)
        cells = [r for r in measured if r['registration'] == registration and r['batch'] == '256'
                 and r['horizon'] == '8' and r['selection'] == 'all']
        if cells:
            assert len(cells) == 2
            by_device = {r['output_device']: r for r in cells}
            cpu, gpu = by_device['cpu'], by_device['cuda']
            assert cpu['host'] == gpu['host'] and cpu['job'] == gpu['job']
            row.update(batch=256, horizon=8, selection='all', source_native_gpu_supported=False,
                parent_torch_threads=1, measured_production_file_hashes_match=cpu['measured_production_file_hashes_match'],
                evidence=cpu['report'])
            for prefix, data in (('cpu', cpu), ('gpu', gpu)):
                for suffix in ('mean', 'sd'):
                    row[f'{prefix}_mse_{suffix}'] = data[f'mse_{suffix}']
                    row[f'ours_{prefix}_fps_{suffix}'] = data[f'ours_fps_{suffix}']
            for mode in ('serial', 'parallel'):
                for suffix in ('mean', 'sd'):
                    row[f'source_cpu_{mode}_fps_{suffix}'] = cpu[f'source_{mode}_fps_{suffix}']
            for suffix in ('mean', 'sd'):
                row[f'source_cpu_cuda_delivery_parallel_fps_{suffix}'] = gpu[f'source_parallel_fps_{suffix}']
            row['ours_cpu_vs_best_source'] = cpu['ours_vs_best_source']
            row['ours_gpu_vs_best_cpu_cuda_delivery'] = gpu['ours_vs_best_source']
        overview.append(row)
    assert Counter(r['route'] for r in overview if r['semantic_status'] == 'VERIFIED' and r['family'] != 'craftax') == {
        'process_when_eligible': 27, 'native_torch': 3, 'local': 5}
    args.output.mkdir(parents=True)
    write_csv(args.output / 'coverage.csv', overview)
    write_csv(args.output / 'measured_online.csv', measured)
    (args.output / 'audit.json').write_text(json.dumps(dict(
        status='AUDITED_INCOMPLETE', scope='Online source results only; offline renderer timings and library controls excluded',
        benchmark_rerun=False, counts=dict(Counter(r['online_source_status'] for r in overview)),
        input_sha256=inputs, generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        measured_rows=len(measured), measured_registrations=len(REGISTRATIONS),
        gpu_vs_native_gpu_rows=0, current_goal_complete=False), indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()

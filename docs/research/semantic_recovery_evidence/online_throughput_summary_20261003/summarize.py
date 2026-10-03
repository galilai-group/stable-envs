"""Reconcile measured components without mixing internal controls into source results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(__doc__)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
E = Path(__file__).parents[1]
inputs = {}
def read(path):
    inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return list(csv.DictReader(path.open()))

configs = []
classic = {'acrobot': 'AcrobotControl-v1', 'cartpole': 'CartPoleControl-v1', 'pendulum': 'PendulumControl-v1',
           'mountain_car': 'MountainCarControl-v0', 'mountain_car_continuous': 'MountainCarContinuousControl-v0'}
for domain, name in classic.items():
    configs.append((name, E/'online_resume_20261003/classic-verified'/f'{domain}.csv', None, 'registration default'))
for domain, name in [('acrobot', 'AcrobotDMControl-v0'), ('pendulum', 'PendulumDMControl-v0'), ('cartpole', 'CartpoleDMControl-v0')]:
    folder = E / ('dmc_cartpole_source_completed_20261003/tables' if domain == 'cartpole' else f'online_eleven_completed_20261003/tables/{domain}')
    configs.append((name, folder/'comparison.csv', folder/'components.csv', '64x64'))
for domain, name in [('two_room', 'TwoRoom-v1'), ('piecewise', 'Piecewise-v0'), ('simple_point_maze', 'SimplePointMaze-v0')]:
    folder = E / ('online_eleven_completed_20261003/tables' if domain == 'simple_point_maze' else 'online_resume_20261003/navigation-verified')
    configs.append((name, folder/f'{domain}.csv', folder/f'{domain}-components.csv', '64x64'))
for suffix in ('', 'Dense', 'Dict'):
    name = f'FetchPickAndPlace{suffix}-v3'
    folder = E/'online_resume_20261003/pick-verified'/name
    configs.append((name, folder/'online.csv', folder/'components.csv', 'no frames measured'))
rows = []
for name, path, component_path, resolution in configs:
    cells = read(path)
    components = read(component_path) if component_path else []
    pick = name.startswith('Fetch')
    for device in ('cpu', 'cuda'):
        def match(r):
            return r.get('device', r.get('output_device')) == device and r['batch'] == '64'
        selected = [r for r in cells if match(r) and r['horizon'] == '8']
        step = next(r for r in selected if r.get('selection', r.get('frames')) == 'none')
        both = None if pick else next(r for r in selected if r['selection'] == 'all')
        render = next((r for r in components if match(r) and r['operation'] == 'render_saved_state'), None)
        raw = next((r for r in components if match(r) and r['operation'] in ('raw_task_step_with_state', 'task_no_state_with_ipc_transfer')), None)
        state = next((r for r in components if match(r) and r['operation'] == 'get_state'), None)
        row = dict(registration='swm/'+name, device=device, batch=64, horizon=8, resolution=resolution,
                   state_history='none', mandatory_final_state=True, combined_frames='none' if pick else 'all',
                   step_only_steps_s_mean=step['ours_steps_s_mean' if pick else 'ours_steps_per_s_mean'],
                   step_only_steps_s_sd=step['ours_steps_s_sd' if pick else 'ours_steps_per_s_sd'],
                   raw_step_scope=raw['operation'] if raw else '', raw_step_steps_s_mean=raw['rate_mean'] if raw else '',
                   raw_step_steps_s_sd=raw['rate_sd'] if raw else '',
                   get_state_states_s_mean=state['rate_mean'] if state else '', get_state_states_s_sd=state['rate_sd'] if state else '',
                   render_only_frames_s_mean=render['rate_mean'] if render else '', render_only_frames_s_sd=render['rate_sd'] if render else '',
                   combined_frames_s_mean=both['ours_fps_mean'] if both else '', combined_frames_s_sd=both['ours_fps_sd'] if both else '',
                   combined_valid_steps_s_mean=both['ours_steps_per_s_mean'] if both else '',
                   combined_valid_steps_s_sd=both['ours_steps_per_s_sd'] if both else '',
                   combined_mse_mean=both['mse_mean'] if both else '', combined_mse_sd=both['mse_sd'] if both else '',
                   job=step['job'], host=step['host'], online_evidence=str(path), component_evidence=str(component_path or ''))
        rows.append(row)
assert len(rows) == 28
a.output.mkdir(parents=True)
with (a.output/'throughput.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
(a.output/'inputs.json').write_text(json.dumps(inputs, indent=2)+'\n')

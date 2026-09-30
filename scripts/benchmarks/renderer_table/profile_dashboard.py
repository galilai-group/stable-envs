"""Standalone sortable view of an exported profiling CSV; no network requests."""

import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("csv", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    with a.csv.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 42
    data = json.dumps(rows).replace("<", "\\u003c")
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Renderer performance · 42 registrations</title>
<style>
body{font:15px system-ui,sans-serif;margin:32px auto;max-width:1300px;padding:0 20px;color:#182238;background:#f7f9fc}h1{font-size:28px;margin-bottom:8px}p{max-width:1000px;line-height:1.5;color:#4c5870}select,input{padding:9px;border:1px solid #bcc7d8;border-radius:5px;background:white;font:inherit;margin:5px}table{border-collapse:collapse;width:100%;background:white}th,td{padding:10px 12px;text-align:left;border-bottom:1px solid #e4e9f0}th{position:sticky;top:0;background:#eaf0f8;font-size:13px}td.number{font-variant-numeric:tabular-nums;white-space:nowrap}td.note{font-size:12px;color:#53617b}small{color:#5c6b83}details{margin:18px 0}a{color:#155ac6}.table{overflow:auto}
</style>
<h1>Renderer performance · 42 registrations</h1>
<p>Fresh Oscar profiles. Choose a rendering stage to find the expensive environments. Values are milliseconds per frame, mean ± sample SD. CPU and GPU jobs can use different host hardware; these are observed costs, not controlled CPU/GPU speedups.</p>
<label>Stage <select id="stage"><option value="public_full_state_b1">Full State → image · batch 1</option><option value="prepare_validate_b1">State validation and preparation · batch 1</option><option value="prepared_render_b1">Prepared input → image · batch 1</option><option value="prepared_batch_render_b256">Prepared input → images · batch 256</option></select></label>
<label>Sort <select id="sort"><option value="gpu">Slowest GPU first</option><option value="cpu">Slowest CPU first</option><option value="env">Environment name</option></select></label>
<input id="filter" placeholder="Filter environments" aria-label="Filter environments">
<p><a href="per_environment_42.csv">42-row CSV</a> · <a href="timings.csv">All stages and backends</a> · <a href="functions.csv">Python call profiles</a> · <a href="README.md">Findings and methods</a></p>
<details><summary>Scope and interpretation</summary><p>Batch 256 uses prepared inputs and excludes full-State preparation. It repeats up to six real saved poses; it does not represent 256 independent simulator instances. Stages are measured separately and should not be added into an exact cost decomposition. Cube uses GPU mjwarp_fidelity; Scene and Walker use OpenGL; their faster alternatives remain in the full CSV. Symbolic Craftax has no RGB renderer. PushT is diagnostic only. FetchPickAndPlace uses reset-only source States. These profiles do not change rendering quality or semantic acceptance.</p><p>Timings use 3–30 warmed samples per probe, with explicit sample counts in the CSV. Every timing outlier is retained. Hover over hardware/backend entries for the host name. Torch traces may not expose external OpenGL/Warp/JAX kernels.</p></details>
<div class="table"><table><thead><tr><th>Environment</th><th>CPU ms/frame ± SD</th><th>GPU ms/frame ± SD</th><th>CPU backend</th><th>GPU backend / hardware</th><th>Scope</th></tr></thead><tbody id="body"></tbody></table></div>
<script>
const rows=__DATA__;
const stage=document.getElementById('stage'), sort=document.getElementById('sort'), filter=document.getElementById('filter');
function value(row,device,stat){const s=row[device+'_'+stage.value+'_'+stat+'_per_frame'];return s===undefined||s===''?null:Number(s);}
function format(row,device){const m=value(row,device,'mean_ms'),s=value(row,device,'sd_ms');return m===null?'—':m.toFixed(m<1?3:2)+' ± '+s.toFixed(m<1?3:2);}
function draw(){const body=document.getElementById('body');body.replaceChildren();const selected=rows.filter(r=>r.env.toLowerCase().includes(filter.value.toLowerCase()));selected.sort((a,b)=>sort.value==='env'?a.env.localeCompare(b.env):(value(b,sort.value,'mean_ms')??-1)-(value(a,sort.value,'mean_ms')??-1));for(const r of selected){const tr=document.createElement('tr');const vals=[r.env,format(r,'cpu'),format(r,'gpu'),r.cpu_backend,r.gpu_backend+' / '+(r.gpu_hardware||'—'),r.scope];vals.forEach((s,i)=>{const td=document.createElement('td');td.textContent=s;if(i===1||i===2)td.className='number';if(i===5)td.className='note';if(i===3)td.title=r.cpu_host+' · '+r.cpu_hardware;if(i===4)td.title=r.gpu_host;tr.append(td);});body.append(tr);}}
for(const el of [stage,sort,filter])el.addEventListener('input',draw);draw();
</script></html>"""
    assert not a.output.exists()
    a.output.write_text(page.replace("__DATA__", data))


if __name__ == "__main__":
    main()

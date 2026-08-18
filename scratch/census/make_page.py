"""Render the merged benchmark into one self-contained HTML page.

Generated rather than hand-written: 196 rows change every time the sweep is rerun, and a page whose
numbers are transcribed by hand is a page whose numbers drift from the JSON they claim to show.
"""
from __future__ import annotations

import json
import statistics as st

RES = "/mnt/vast/home/randall.balestriero/code/stable-envs/examples/results/decode"
OUT = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/census/stable-envs-throughput.html"

d = json.load(open(f"{RES}/ALL.json"))
rows = d["rows"]
for r in rows:
    r["cores"] = r["gpu"] / r["dec"]
    r["vs"] = (r["gpu"] / r["cpu"]) if r["cpu"] else None
rows.sort(key=lambda r: -r["cores"])

cores = [r["cores"] for r in rows]
vs = [r["vs"] for r in rows if r["vs"]]
fileratio = sorted(r["file"] / r["dec"] for r in rows if r["file"])
n_default = sum(1 for r in rows if r["cpu"])
n_below = sum(1 for c in cores if c < 1)
tot_state = sum(r["state_b"] for r in rows if r["state_b"])
tot_raw = sum(r["raw_b"] for r in rows)

fams: dict[str, list] = {}
for r in rows:
    fams.setdefault(r["family"], []).append(r)


def rng(vals, fmt="{:,.0f}"):
    vals = [v for v in vals if v]
    if not vals:
        return "&mdash;"
    lo, hi = min(vals), max(vals)
    a, b = fmt.format(lo), fmt.format(hi)
    return a if a == b else f"{a}&ndash;{b}"


fam_rows = ""
for fam, rs in sorted(fams.items(), key=lambda kv: -max(r["cores"] for r in kv[1])):
    shapes = sorted({r["shape"] for r in rs})
    fam_rows += (
        f"<tr><td class='nm'>{fam}</td><td class='n'>{len(rs)}</td>"
        f"<td class='n dim'>{shapes[0] if len(shapes) == 1 else shapes[0] + '&hellip;'}</td>"
        f"<td class='n'>{rng([r['cpu'] for r in rs])}</td>"
        f"<td class='n ours'>{rng([r['gpu'] for r in rs])}</td>"
        f"<td class='n'>{rng([r['dec'] for r in rs])}</td>"
        f"<td class='n hero'>{rng([r['cores'] for r in rs], '{:,.1f}')}&times;</td></tr>")

data = [{"n": r["name"], "f": r["family"], "s": r["shape"], "c": r["cpu"], "g": r["gpu"],
         "b": r["batch"], "d": r["dec"], "k": r["cores"], "v": r["vs"],
         "sb": r["state_b"], "jb": r["jpeg_b"]} for r in rows]

# ---- the 256px / batch-1024 comparison against a real Lance corpus -------------------------------
import collections
import glob

LANCE = json.load(open(f"{RES}/../storage/lance.json"))["envs"]
GPU256 = {}
for _p in glob.glob(f"{RES}/../storage/storage_*.json"):
    for _k, _v in json.load(open(_p))["envs"].items():
        if isinstance(_v.get("gpu"), dict) and _v["gpu"].get("fps"):
            GPU256[_k] = _v["gpu"]["fps"]
FAMOF = {r["name"]: r["family"] for r in rows}
_by = collections.defaultdict(list)
_paths = collections.defaultdict(list)
for _k, _g in GPU256.items():
    _L = LANCE.get(_k)
    if not _L:
        continue
    _by[FAMOF.get(_k, "other")].append((_g / _L["local_1"]["fps"], _g / _L["local_n"]["fps"],
                                        _g / _L["local_gpu"]["fps"]))
    for _t in ("local_1", "shared_1", "local_n", "local_gpu"):
        _paths[_t].append(_L[_t]["fps"])
_all = [v for vs in _by.values() for v in vs]
store_rows = ""
for _f in sorted(_by, key=lambda f: -st.median([x[1] for x in _by[f]])):
    _v = _by[_f]
    _m = st.median([x[1] for x in _v])
    store_rows += (f"<tr class='{'' if _m >= 1 else 'below'}'><td class='nm'>{_f}</td>"
                   f"<td class='n'>{len(_v)}</td>"
                   f"<td class='n'>{st.median([x[0] for x in _v]):,.1f}&times;</td>"
                   f"<td class='n hero'>{_m:,.2f}&times;</td>"
                   f"<td class='n'>{st.median([x[2] for x in _v]):,.2f}&times;</td></tr>")
path_rows = ""
for _t, _lbl in (("local_1", "node-local NVMe, 1 core"), ("shared_1", "shared filesystem, 1 core"),
                 ("local_n", "node-local, 8 worker processes"),
                 ("local_gpu", "node-local, nvJPEG decode on GPU")):
    _v = _paths[_t]
    path_rows += (f"<tr><td class='nm'>{_lbl}</td><td class='n'>{st.median(_v):,.0f}</td>"
                  f"<td class='n dim'>{min(_v):,.0f}</td><td class='n dim'>{max(_v):,.0f}</td></tr>")
n_win = sum(1 for x in _all if x[1] >= 1)

HTML = f"""<title>stable-envs &middot; one frame, three ways</title>
<style>
:root {{
  --paper:#f4f6f5; --surface:#ffffff; --ink:#0e1518; --muted:#55666b; --rule:#dae1e0;
  --rule-soft:#e9eeed; --accent:#0a6b63; --accent-soft:#e2f0ee; --warn:#a63b1d;
  --warn-soft:#f6e6df; --shadow:0 1px 2px rgba(14,21,24,.05);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --paper:#0c1214; --surface:#121a1d; --ink:#e4ecea; --muted:#8ea2a6; --rule:#222f32;
    --rule-soft:#1a2427; --accent:#5ad3c2; --accent-soft:#123430; --warn:#e5906c;
    --warn-soft:#3a2118; --shadow:0 1px 2px rgba(0,0,0,.3);
  }}
}}
:root[data-theme="dark"] {{
  --paper:#0c1214; --surface:#121a1d; --ink:#e4ecea; --muted:#8ea2a6; --rule:#222f32;
  --rule-soft:#1a2427; --accent:#5ad3c2; --accent-soft:#123430; --warn:#e5906c;
  --warn-soft:#3a2118; --shadow:0 1px 2px rgba(0,0,0,.3);
}}
* {{ box-sizing:border-box; }}
body {{
  background:var(--paper); color:var(--ink); margin:0;
  font:400 16px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:1180px; margin:0 auto; padding:clamp(2rem,5vw,4.5rem) clamp(1rem,4vw,2.5rem) 6rem; }}
.mono {{ font-family:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace; }}
.eyebrow {{
  font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace; font-size:.72rem;
  letter-spacing:.14em; text-transform:uppercase; color:var(--muted); margin:0 0 1.1rem;
}}
h1 {{ font-size:clamp(2rem,4.6vw,3.1rem); line-height:1.08; letter-spacing:-.025em;
     margin:0 0 1.1rem; text-wrap:balance; font-weight:600; }}
h2 {{ font-size:1.32rem; letter-spacing:-.012em; margin:0 0 .35rem; font-weight:600; }}
h2 + .sub {{ margin:0 0 1.4rem; }}
.sub {{ color:var(--muted); max-width:66ch; }}
.lede {{ font-size:1.08rem; max-width:64ch; color:var(--muted); margin:0 0 2.6rem; }}
.lede strong {{ color:var(--ink); font-weight:600; }}
section {{ margin-top:4rem; }}

.figs {{ display:grid; gap:1px; background:var(--rule); border:1px solid var(--rule);
        border-radius:10px; overflow:hidden; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); }}
.fig {{ background:var(--surface); padding:1.15rem 1.2rem 1.25rem; }}
.fig .v {{ font-family:ui-monospace,"SF Mono",Menlo,monospace; font-size:1.7rem;
          letter-spacing:-.03em; font-variant-numeric:tabular-nums; display:block; line-height:1.15; }}
.fig .k {{ font-size:.79rem; color:var(--muted); display:block; margin-top:.4rem; }}
.fig.a .v {{ color:var(--accent); }}
.fig.w .v {{ color:var(--warn); }}

.defn {{ display:grid; gap:1px; background:var(--rule); border:1px solid var(--rule);
         border-radius:10px; overflow:hidden; grid-template-columns:repeat(auto-fit,minmax(250px,1fr));
         margin:0 0 2rem; }}
.defn > div {{ background:var(--surface); padding:1.1rem 1.2rem; }}
.defn b {{ display:block; font-family:ui-monospace,Menlo,monospace; font-size:.8rem;
           letter-spacing:.08em; text-transform:uppercase; margin-bottom:.45rem; }}
.defn p {{ margin:0; font-size:.88rem; color:var(--muted); }}
.defn .ours b {{ color:var(--accent); }}

.scroll {{ overflow-x:auto; border:1px solid var(--rule); border-radius:10px; background:var(--surface);
           box-shadow:var(--shadow); }}
table {{ border-collapse:collapse; width:100%; font-size:.855rem; }}
th, td {{ padding:.5rem .75rem; border-bottom:1px solid var(--rule-soft); white-space:nowrap; }}
thead th {{ position:sticky; top:0; background:var(--surface); text-align:right; font-weight:600;
            font-size:.73rem; letter-spacing:.06em; text-transform:uppercase; color:var(--muted);
            border-bottom:1px solid var(--rule); z-index:2; }}
thead th:first-child, thead th:nth-child(2) {{ text-align:left; }}
td.n {{ text-align:right; font-family:ui-monospace,"SF Mono",Menlo,monospace;
        font-variant-numeric:tabular-nums; }}
td.nm {{ font-weight:500; }}
td.dim {{ color:var(--muted); }}
td.ours {{ color:var(--accent); }}
td.hero {{ font-weight:600; }}
tbody tr:hover {{ background:var(--accent-soft); }}
tbody tr:last-child td {{ border-bottom:none; }}
tr.below td.hero {{ color:var(--warn); }}
tr.below td.bar i {{ background:var(--warn); }}
td.bar {{ width:130px; padding-right:1rem; }}
td.bar span {{ display:block; position:relative; height:7px; border-radius:2px;
               background:var(--rule-soft); }}
td.bar i {{ position:absolute; left:0; top:0; bottom:0; background:var(--accent); border-radius:2px; }}
td.bar span::after {{ content:""; position:absolute; top:-3px; bottom:-3px; width:1px;
                      background:var(--muted); opacity:.45; left:var(--be); }}

.sortable th {{ cursor:pointer; user-select:none; }}
.sortable th[aria-sort]::after {{ content:" \\2191"; opacity:.5; }}
.sortable th[aria-sort="descending"]::after {{ content:" \\2193"; }}
.controls {{ display:flex; gap:.6rem; flex-wrap:wrap; align-items:center; margin:0 0 1rem; }}
input[type=search], select {{
  font:inherit; font-size:.85rem; padding:.42rem .7rem; border:1px solid var(--rule);
  border-radius:7px; background:var(--surface); color:var(--ink); min-width:180px;
}}
input:focus-visible, select:focus-visible, th:focus-visible {{
  outline:2px solid var(--accent); outline-offset:2px; }}
.count {{ font-family:ui-monospace,Menlo,monospace; font-size:.8rem; color:var(--muted); }}

.notes {{ border-top:1px solid var(--rule); padding-top:1.6rem; }}
.notes li {{ margin:0 0 .8rem; color:var(--muted); max-width:74ch; }}
.notes li strong {{ color:var(--ink); }}
code {{ font-family:ui-monospace,Menlo,monospace; font-size:.86em; background:var(--rule-soft);
        padding:.1em .35em; border-radius:4px; }}
.foot {{ margin-top:3.5rem; padding-top:1.4rem; border-top:1px solid var(--rule);
         font-size:.8rem; color:var(--muted); }}
@media (prefers-reduced-motion:no-preference) {{ tbody tr {{ transition:background .12s; }} }}
</style>

<div class="wrap">
<p class="eyebrow">stable-envs &middot; measured on one H200 + Intel Xeon Platinum 8558</p>
<h1>One frame, three ways</h1>
<p class="lede">Every number on this page is frames per second for the same transformation,
<span class="mono">state&nbsp;&rarr;&nbsp;uint8 frame</span>, at each environment's own native resolution,
across <strong>{len(rows)} environments</strong>. The question is not whether a GPU renderer beats the
env's own; it is whether <strong>regenerating a frame beats reading one you stored</strong> &mdash; and
for {n_below} of {len(rows)} environments the answer is no.</p>

<div class="defn">
  <div><b>default</b><p>The env's own official renderer, one frame at a time &mdash; the only entry
  point it offers.</p></div>
  <div class="ours"><b>ours</b><p>The GPU path at its best batch size.</p></div>
  <div><b>stored</b><p>One CPU core decoding <em>this env's own frame</em> as JPEG&nbsp;q90, already in
  memory. No filesystem, no network &mdash; the case most favourable to a stored dataset.</p></div>
</div>

<div class="figs">
  <div class="fig"><span class="v">{len(rows)}</span><span class="k">environments measured</span></div>
  <div class="fig a"><span class="v">{st.median(cores):,.1f}&times;</span>
    <span class="k">median cores of decode replaced</span></div>
  <div class="fig"><span class="v">{st.geometric_mean(cores):,.1f}&times;</span>
    <span class="k">geometric mean</span></div>
  <div class="fig"><span class="v">{max(cores):,.0f}&times;</span><span class="k">best (GriddlySokoban)</span></div>
  <div class="fig w"><span class="v">{n_below}</span>
    <span class="k">envs where reading wins</span></div>
  <div class="fig"><span class="v">{st.median(vs):,.0f}&times;</span>
    <span class="k">median vs the official renderer</span></div>
</div>

<section>
<h2>Against a real dataset, at one operating point</h2>
<p class="sub">Everything above is each environment at its native size against one CPU core. This is the
harder question: <strong>256&times;256, batch 1024</strong> for every environment, against the fastest
disk-based setup in current use &mdash; JPEG&nbsp;q95 blobs in an episode-contiguous Lance table, the
layout <span class="mono">stable_worldmodel</span> writes, chunk-fetched and batch-decoded through
libjpeg-turbo. Measured for all {len(_all)} environments in one process on one node, so the only thing
varying is compressed bytes per frame.</p>

<div class="figs" style="margin-bottom:1.6rem">
  <div class="fig a"><span class="v">{st.median([x[0] for x in _all]):,.1f}&times;</span>
    <span class="k">median vs 1 CPU core</span></div>
  <div class="fig{'' if st.median([x[1] for x in _all]) >= 1 else ' w'}">
    <span class="v">{st.median([x[1] for x in _all]):,.2f}&times;</span>
    <span class="k">median vs 8 worker processes</span></div>
  <div class="fig{'' if st.median([x[2] for x in _all]) >= 1 else ' w'}">
    <span class="v">{st.median([x[2] for x in _all]):,.2f}&times;</span>
    <span class="k">median vs nvJPEG on GPU</span></div>
  <div class="fig"><span class="v">{n_win}/{len(_all)}</span>
    <span class="k">envs still ahead of 8 workers</span></div>
</div>

<p class="sub">The split is the same axis as the parity story: environments whose frame is a
<strong>gather from a constant atlas</strong> win by one to three orders of magnitude; environments that
<strong>ray-cast a 3-D scene per pixel</strong> lose to a well-parallelised loader at this resolution.
Rows below the break-even line are marked.</p>
<div class="scroll"><table>
<thead><tr><th>family</th><th>envs</th><th>vs 1 core</th><th>vs 8 workers</th><th>vs nvJPEG</th></tr></thead>
<tbody>{store_rows}</tbody></table></div>

<h2 style="margin-top:2.2rem">The four read paths</h2>
<p class="sub">Median across all environments. The shared filesystem matching node-local NVMe is the
finding: a stored-frame pipeline is decode-bound, not IO-bound, at these rates
(44&ndash;473&nbsp;MB/s).</p>
<div class="scroll"><table>
<thead><tr><th>read path</th><th>median f/s</th><th>min</th><th>max</th></tr></thead>
<tbody>{path_rows}</tbody></table></div>
</section>

<section>
<h2>By family</h2>
<p class="sub">Ranges span the environments in each family. Sorted by the best ratio the family reaches.</p>
<div class="scroll"><table>
<thead><tr><th>family</th><th>envs</th><th>frame</th><th>default f/s</th><th>ours f/s</th>
<th>stored f/s/core</th><th>cores replaced</th></tr></thead>
<tbody>{fam_rows}</tbody></table></div>
</section>

<section>
<h2>Every environment</h2>
<p class="sub">Click a column to sort. The bar is <span class="mono">cores replaced</span> on a log scale;
the tick marks break-even, where regenerating costs exactly what reading costs.</p>
<div class="controls">
  <input type="search" id="q" placeholder="filter by name or family&hellip;" aria-label="Filter environments">
  <select id="fam" aria-label="Filter by family"><option value="">all families</option></select>
  <span class="count" id="count"></span>
</div>
<div class="scroll"><table class="sortable" id="t">
<thead><tr>
  <th data-k="n" aria-sort="none">env</th><th data-k="f">family</th><th data-k="s">frame</th>
  <th data-k="c">default</th><th data-k="g">ours</th><th data-k="b">batch</th>
  <th data-k="d">stored /core</th><th data-k="k" aria-sort="descending">cores</th>
  <th data-k="v">vs default</th><th data-k="sb">state B</th><th data-k="jb">jpeg B</th>
</tr></thead>
<tbody id="tb"></tbody></table></div>
</section>

<section class="notes">
<h2>How to read this</h2>
<ul>
<li><strong>The load half of &ldquo;load and decode&rdquo; barely exists.</strong> Reading a warm local
JPEG file and decoding it runs at {fileratio[0]:.2f}&ndash;{fileratio[-1]:.2f}&times; the speed of
decoding the same bytes already in memory (median {fileratio[len(fileratio) // 2]:.2f}&times;). A
stored-frame pipeline is decode-bound, not IO-bound &mdash; so the in-memory column, chosen because it
favours datasets, is very nearly the only column. This says nothing about a dataset that must cross a
network; that would do worse.</li>
<li><strong>The two comparisons are not equally reproducible.</strong> The GPU column repeats across
nodes to under a percent &mdash; one kernel on a device the job owns. The <em>default</em> column is a
single-threaded Python renderer sharing a node, and the same measurement has moved by more than
100&times; with load. Read it as an order of magnitude.</li>
<li><strong>{len(rows) - n_default} environments have no <em>default</em> number</strong> because the
reference package is not installed in the interpreter that owns them (<code>highway_env</code>,
<code>crafter</code>, <code>craftax</code>, <code>minihack</code>). That is an absence, not a zero. An
earlier pass showed 49 blanks; 41 of those were a bug in the benchmark's own reference bridge, which
reached for <code>ref.unwrapped</code> on references that are not gymnasium envs and threw away a
timing it had already taken.</li>
<li><strong>One GPU against one CPU core</strong> is the right comparison for &ldquo;store or
regenerate on the node I have&rdquo;, and the wrong one for a per-watt or per-dollar argument.</li>
<li><strong>Storage, for scale.</strong> Summed over all {len(rows)} envs, the state that determines a
frame is {tot_state:,}&nbsp;B against {tot_raw:,}&nbsp;B of raw pixels &mdash; a factor of
{tot_raw / tot_state:,.0f}, before any compression on either side.</li>
<li><strong>Measured across 15 interpreters</strong>, because no single one can import every family:
<code>robocasa</code> pins mujoco&nbsp;3.3.1 while <code>mujoco_warp</code> needs&nbsp;&ge;&nbsp;3.11,
and jaxmarl and torch do not co-install. The widest single interpreter reaches 147 of {len(rows)}.</li>
</ul>
<p class="foot">Generated from <span class="mono">examples/results/decode/ALL.json</span> by
<span class="mono">scratch/census/make_page.py</span>. Reproduce with
<span class="mono">examples/bench_vs_decode.py</span>.</p>
</section>
</div>

<script>
const D = {json.dumps(data)};
const tb = document.getElementById('tb'), q = document.getElementById('q'),
      famSel = document.getElementById('fam'), count = document.getElementById('count');
[...new Set(D.map(r => r.f))].sort().forEach(f => {{
  const o = document.createElement('option'); o.value = o.textContent = f; famSel.append(o);
}});
const fmt = (v, dp = 0) => v == null ? '&mdash;'
  : v.toLocaleString('en-US', {{minimumFractionDigits: dp, maximumFractionDigits: dp}});
// Log scale: the ratios span 0.02 to 3648, five orders of magnitude. A linear bar would render every
// row below ~100 as an empty cell, which is exactly the range where the interesting rows live.
const LO = Math.log10(0.02), HI = Math.log10(4000), BE = (0 - LO) / (HI - LO) * 100;
const pct = v => Math.max(1.5, Math.min(100, (Math.log10(v) - LO) / (HI - LO) * 100));
let sortKey = 'k', desc = true;

function draw() {{
  const term = q.value.trim().toLowerCase(), fam = famSel.value;
  const rows = D.filter(r => (!fam || r.f === fam) &&
      (!term || r.n.toLowerCase().includes(term) || r.f.toLowerCase().includes(term)));
  rows.sort((a, b) => {{
    const x = a[sortKey], y = b[sortKey];
    if (x == null) return 1;
    if (y == null) return -1;
    const c = typeof x === 'string' ? x.localeCompare(y) : x - y;
    return desc ? -c : c;
  }});
  tb.innerHTML = rows.map(r => `<tr class="${{r.k < 1 ? 'below' : ''}}">
    <td class="nm">${{r.n}}</td><td class="dim">${{r.f}}</td><td class="n dim">${{r.s}}</td>
    <td class="n">${{fmt(r.c)}}</td><td class="n ours">${{fmt(r.g)}}</td>
    <td class="n dim">${{fmt(r.b)}}</td><td class="n">${{fmt(r.d)}}</td>
    <td class="n hero">${{fmt(r.k, r.k < 10 ? 2 : 0)}}&times;</td>
    <td class="bar"><span style="--be:${{BE}}%"><i style="width:${{pct(r.k)}}%"></i></span></td>
    <td class="n dim">${{r.v == null ? '&mdash;' : fmt(r.v) + '&times;'}}</td>
    <td class="n dim">${{fmt(r.sb)}}</td><td class="n dim">${{fmt(r.jb)}}</td></tr>`).join('');
  count.textContent = `${{rows.length}} of ${{D.length}} shown`;
}}

document.querySelectorAll('#t thead th[data-k]').forEach(th => {{
  th.tabIndex = 0;
  const go = () => {{
    const k = th.dataset.k;
    desc = k === sortKey ? !desc : (k !== 'n' && k !== 'f' && k !== 's');
    sortKey = k;
    document.querySelectorAll('#t thead th').forEach(o => o.removeAttribute('aria-sort'));
    th.setAttribute('aria-sort', desc ? 'descending' : 'ascending');
    draw();
  }};
  th.addEventListener('click', go);
  th.addEventListener('keydown', e => {{ if (e.key === 'Enter' || e.key === ' ') {{ e.preventDefault(); go(); }} }});
}});
q.addEventListener('input', draw);
famSel.addEventListener('change', draw);
draw();
</script>
"""

# The bar column has no header cell of its own; it rides along with `cores`. Inserting one keeps the
# header count equal to the body's, which is what lets the sticky header line up.
HTML = HTML.replace('<th data-k="k" aria-sort="descending">cores</th>',
                    '<th data-k="k" aria-sort="descending">cores</th><th></th>')
open(OUT, "w").write(HTML)
print(f"wrote {OUT}  ({len(HTML) / 1024:.0f} kB, {len(rows)} rows, {n_default} with a default column)")

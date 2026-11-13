"""MkDocs hooks that resolve links to repository source files and expand the environment tables.

Only the built site changes: linked source files become downloadable assets and linked
directories get generated file indexes. Strict link and anchor validation stays enabled.
Each ``<!-- env: Name -->`` line becomes the clips and tables of that class, read from
``docs/environments/_data/<Name>.json``.
"""

import json
import posixpath
import re
from pathlib import Path

from mkdocs.structure.files import File

_LINK = re.compile(r"(?P<open>\]\()(?P<target>[^\s)]+)(?P<close>\))")
_rewrites = {}


def on_files(files, config):
    """Publish linked package sources and generate indexes for linked evidence directories."""
    _rewrites.clear()
    docs = Path(config.docs_dir).resolve()
    root = docs.parent
    generated = set()
    for file in list(files):
        if not file.is_documentation_page():
            continue
        for match in _LINK.finditer(file.content_string):
            target = match["target"]
            if ":" in target or "#" in target or target.startswith("/"):
                continue
            resolved = (docs / file.src_uri).parent.joinpath(target).resolve()
            destination = None
            if resolved.is_file() and resolved.is_relative_to(root / "stable_envs"):
                destination = "_source/" + resolved.relative_to(root).as_posix()
                if destination not in generated:
                    files.append(File.generated(config, destination, abs_src_path=str(resolved)))
            elif target.endswith("/") and resolved.is_dir() and resolved.is_relative_to(docs):
                destination = resolved.relative_to(docs).as_posix() + "/index.md"
                if files.get_file_from_path(destination) is None and destination not in generated:
                    entries = [p for p in sorted(resolved.iterdir()) if p.is_file()]
                    content = "# Evidence directory\n\nGenerated index; original evidence files are unchanged.\n\n"
                    content += "".join(f"- [{p.name}]({p.name})\n" for p in entries)
                    files.append(File.generated(config, destination, content=content))
            if destination is not None:
                generated.add(destination)
                _rewrites[file.src_uri, target] = posixpath.relpath(destination, posixpath.dirname(file.src_uri))
    return files


def on_page_markdown(markdown, page, config, **kwargs):
    """Expand the environment sections and point the page's links at the published files."""

    def replace(match):
        target = _rewrites.get((page.file.src_uri, match["target"]), match["target"])
        return match["open"] + target + match["close"]

    return _LINK.sub(replace, _env_sections(markdown, page, config))


_ENV = re.compile(r"^<!-- env: (?P<name>\w+) -->$", re.M)
_MEDIA = "assets/media/envs"
_COLLAPSE = 12
_DEVICE = {"cpu": "CPU", "cuda": "CUDA"}
_MISSING = re.compile(
    r"ModuleNotFoundError: No module named '(?P<module>[\w.]+)'|.*'stable-envs\[(?P<extra>[\w-]+)\]'"
)


def _num(v):
    v = float(v)
    if v != v:
        return "NaN"
    if v in (float("inf"), float("-inf")):
        return "∞" if v > 0 else "−∞"
    return str(int(v)) if v == int(v) else f"{v:g}"


def _flat(v):
    return v if isinstance(v, list) else [v]


def _shape(shape):
    return "(" + ", ".join(map(str, shape)) + ("," if len(shape) == 1 else "") + ")"


def _bounds(low, high):
    lows, highs = _flat(low), _flat(high)
    if len(lows) == 1 and len(highs) == 1:
        return f"[{_num(lows[0])}, {_num(highs[0])}]"
    if max(len(lows), len(highs)) <= 3:
        return f"[{', '.join(map(_num, lows))}] to [{', '.join(map(_num, highs))}]"
    return f"per element, within [{_num(min(map(float, lows)))}, {_num(max(map(float, highs)))}]"


def _value(v):
    if v is None:
        return ""
    if isinstance(v, list):
        return f"[{', '.join(map(_num, v))}]" if len(v) <= 4 else f"{len(v)} values"
    return _num(v)


def _action(a):
    if a is None:
        return "none (renderer only)"
    text = f"`{a['dtype']}` {_shape(a['shape'])}"
    if "n" in a:
        return f"{text}, {a['n']} choices" + (" per element" if a["shape"] not in ([], [1]) else "")
    if a.get("low") is not None and a.get("high") is not None:
        text += ", " + _bounds(a["low"], a["high"])
    return text


def _video(page_url, name, clip, label):
    base = posixpath.dirname(page_url.rstrip("/") + "/index.html") if page_url else ""
    src = posixpath.relpath(f"{_MEDIA}/{name}/{clip['file']}", base or ".")
    return (
        f'<figure><video autoplay loop muted playsinline width="{clip["width"]}" height="{clip["height"]}" '
        f'src="{src}" aria-label="{name}: {label}"></video><figcaption>{label}</figcaption></figure>'
    )


def _variation_table(rows):
    table = ["| Path | Type | Bounds or choices | Default | Sampled at reset |", "|---|---|---|---|---|"]
    for row in rows:
        if "n" in row:
            kind, bounds = row["type"], f"{row['n']} choices from {row['start']}"
        else:
            kind, bounds = f"{row['type']} {_shape(row['shape'])}", _bounds(row["low"], row["high"])
        reset = "yes" if row["reset"] else ""
        table.append(f"| `{row['path']}` | {kind} | {bounds} | {_value(row['default'])} | {reset} |")
    title = f"Variation space ({len(rows)} leaves)"
    if len(rows) > _COLLAPSE:
        return [f'??? info "{title}"', ""] + ["    " + line for line in table]
    return [f"**{title}**", ""] + table


def _env_section(name, data, page_url):
    out = []
    media = data.get("media") or {}
    clips = [_video(page_url, name, media["rollout"], "Random actions")] if media.get("rollout") else []
    if media.get("variation"):
        clips.append(_video(page_url, name, media["variation"], "Sampled Variation (top left: default)"))
    if clips:
        out += ['<div class="env-media">', *clips, "</div>", ""]
    h, w, c = data["frame_shape"]
    usable = []
    for device, status in (data.get("devices") or {}).items():
        missing = _MISSING.match(status)
        if status == "ok":
            usable.append(_DEVICE.get(device, device))
        elif missing:
            need = f"`{missing['module']}`" if missing["module"] else f"the `{missing['extra']}` extra"
            usable.append(f"{_DEVICE.get(device, device)} (needs {need})")
    rows = [
        ("Reference", f"`{data['reference_id']}`" if data["reference_id"] else "none"),
        ("Physics", data.get("physics") or ""),
        ("Devices", ", ".join(usable)),
        ("Requires CUDA", "yes" if data["requires_cuda"] else "no"),
        ("Renderer on CPU", data["renderers"]["cpu"]),
        ("Renderer on CUDA", data["renderers"]["cuda"]),
        ("Default frame", f"{h} × {w} × {c}, `uint8`"),
        ("Native size", f"{data['native_size']} px, the only size" if data.get("native_size") else ""),
        ("Action", _action(data["action"])),
    ]
    if media.get("device"):
        device = _DEVICE.get(media["device"].split(":")[0], media["device"])
        rows.append(("Clips", f"rendered on {device} at {media['render_size']} px, then resized"))
    out += ["| Property | Value |", "|---|---|", *(f"| {k} | {v} |" for k, v in rows if v), ""]
    out += [f"**State** ({len(data['state'])} top-level keys)", "", "| Key | Shape | dtype |", "|---|---|---|"]
    for row in data["state"]:
        if "leaves" in row:
            shape, dtype = f"{row['leaves']} leaves, {row['numel']} values", ", ".join(row["dtypes"])
        else:
            shape, dtype = _shape(row["shape"]), row["dtype"]
        out.append(f"| `{row['key']}` | {shape} | {dtype} |")
    out.append("")
    if data["variation"]:
        out += _variation_table(data["variation"])
    else:
        out.append("This class has no Variation leaves.")
    out.append("")
    return "\n".join(out)


def _env_sections(markdown, page, config):
    """Expand each ``<!-- env: Name -->`` line into the media and tables of that class."""
    data = Path(config.docs_dir) / "environments" / "_data"

    def replace(match):
        path = data / f"{match['name']}.json"
        if not path.is_file():
            raise FileNotFoundError(f"{page.file.src_uri}: no snapshot {path}; run scripts/gen_env_docs.py")
        return _env_section(match["name"], json.loads(path.read_text()), page.url)

    return _ENV.sub(replace, markdown)

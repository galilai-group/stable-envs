"""Resolve archived repository/file-directory links without editing hashed evidence.

Only the built site is changed. Source files become downloadable assets and
linked evidence directories receive generated file indexes. Strict link and
anchor validation remains enabled.
"""

import posixpath
import re
from pathlib import Path

from mkdocs.structure.files import File

_LINK = re.compile(r"(?P<open>\]\()(?P<target>[^\s)]+)(?P<close>\))")
_rewrites = {}


def on_files(files, config):
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


def on_page_markdown(markdown, page, **kwargs):
    def replace(match):
        target = _rewrites.get((page.file.src_uri, match["target"]), match["target"])
        return match["open"] + target + match["close"]

    return _LINK.sub(replace, markdown)

"""Generate the final Step 6 report from the Step 6 artifacts, and from nothing else.

    python scripts/regression/optimization/report.py --out docs/research/step6_optimization.md

Same discipline as Step 4 and Step 5: every number on the page is read out of a file under
`tests/fixtures/regression/join/step6/` (or out of the frozen Step 3/Step 4 oracles), every cell names the file it
came from, and the page states what was NOT done as plainly as what was. Section 25 of the
coordinator brief asks for exactly that and adds one instruction that shapes this whole file: *do
not manually transcribe counts if they can be generated*. So no count here is typed. The 49-row
configuration table, the four headline counts, the "what Step 6 did not do" list and the semantic
assertion are all computed, and this module contains no measured number of its own.

Three design rules earn their keep and are worth stating before the code:

**The artifact wins, and the disagreement is printed.** The dashboard is hand-maintained prose about
machine-written files. Where the two disagree the page prints the artifact's value and records the
disagreement in a section of its own rather than silently choosing. A report that quietly reconciles
its sources is a report whose sources have stopped being checkable.

**Unknown shapes are listed, never dropped and never fatal.** Eight workstreams wrote JSON in eight
shapes and more will arrive; a generator that crashes on the ninth is a generator that gets replaced
by hand-editing. Every JSON under the results tree is read inside a guard, classified by its keys,
and anything unrecognised is named in the "artifacts not summarised" appendix with its path — which
is a to-do list for the next run, not a silence.

**No ladder is re-implemented.** A re-measured tier is Step 3's own evaluator
(`scripts/regression/semantic/certify.py:classify`) applied to a Step 6 shard, exactly as
`scratch/step6/tilemap/tier.py` does for one row. The two rules that are easy to get wrong by hand —
a failed `cores` conjunct walks DOWN from the rung the ratio picked, and a failed `BATCH_SCALES`
caps at `GPU_PARITY` — are the two a second implementation would get wrong.

Re-running after another merge needs no edit here: the workstreams come from the dashboard, the
merge commits from `git log`, the tiers from whatever tier/sweep artifacts exist at the time, and
the appendix names anything new that this file does not yet understand.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for cand in (p, *p.parents):
        if (cand / "pyproject.toml").is_file():
            return cand
    raise RuntimeError("repo root not found")


ROOT = _repo_root()

#: The role whose performance the frozen contract actually requires. Step 2 §4.2: "The configuration
#: being certified must be the registered SWM default", and §12.2 puts `ENVIRONMENT_DONE` on that
#: default alone. A render-parameter-space cell is certified for CORRECTNESS — it exists to prove the
#: renderer answers the whole parameter space, not to carry a throughput obligation — so counting one
#: as a performance blocker invents an obligation the contract never wrote down. Correctness
#: accounting stays over all 49; performance completion is over these.
OWES_PERFORMANCE = "registered_default"

#: Where the normative prose and the machine-readable bar live. Both are read, neither is restated:
#: the minimum tiers come out of `ladder.environment_done_min_tiers` in the frozen spec, so this file
#: contains no bound of its own.
SPEC_JSON = "tests/fixtures/regression/fidelity/certification.json"
SPEC_MD = "docs/research/certification.md"

#: Cluster paths appear inside artifacts (`shard`, `replay`, `states_source`, `before_dir`). They
#: must never reach the page: `/oscar/scratch/sboughan/...` contains the literal `scratch/...` that
#: `tests/unit/test_repo_contracts.py` extracts as a citation, so printing one would assert that a
#: directory on Oscar is a file in this repository and break the citation contract. Historical
#: `scratch/...` provenance strings in frozen JSON are the same class of claim now that `scratch/`
#: is off the product tree: keep them in the immutable record, do not present them as live paths.
#: Scrubbed on the way out, once, over the whole document.
_CLUSTER = re.compile(r"(?:\$SCR|/oscar)[\w./$-]*")
_HISTORICAL_SCRATCH = re.compile(r"(?<![\w./])scratch/[A-Za-z0-9_./:-]+")


def scrub(x) -> str:
    text = _CLUSTER.sub("<cluster path>", str(x))
    return _HISTORICAL_SCRATCH.sub("<historical provenance>", text)


# ------------------------------------------------------------------------------ git


def git(repo: Path, *args: str) -> str | None:
    """``git -C repo args...``, or None when this is not a checkout (the Oscar copies have no .git)."""
    try:
        p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=30)
    except Exception:
        return None
    return p.stdout.strip() if p.returncode == 0 else None


def merge_commits(repo: Path, ref: str = "HEAD") -> dict[str, list[tuple[str, str]]]:
    """``{workstream: [(sha, subject), ...]}`` for every ``Merge step6/<ws>`` reachable from ``ref``.

    Back-merges (``Merge branch 'step6/integration' into step6/<ws>``) run the other way and are not
    the commit section 25 asks for, so only subjects that START with ``Merge step6/<ws>`` count, and
    ``integration`` is never a workstream. This is read from git rather than from the dashboard on
    purpose: a workstream that reports itself merged and has no merge commit here is a finding.
    """
    out: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for line in (git(repo, "log", "--merges", "--format=%H%x00%s", ref) or "").splitlines():
        sha, _, subject = line.partition("\0")
        m = re.match(r"Merge step6/([A-Za-z0-9_.-]+)", subject)
        if m and m.group(1) != "integration":
            out[m.group(1)].append((sha, subject))
    return out


class CommitOrder:
    """Which of two stamped measurements is the CURRENT one, on the tree being reported.

    A workstream re-runs a certification and commits the new record beside the old one; both are
    evidence, only one is the verdict. "Newest" is decided by **ancestry on the reported tree** —
    `git merge-base --is-ancestor` — because a timestamp is a property of the machine that wrote the
    file and ancestry is a property of the history the page describes. Two records on branches that
    never saw each other are incomparable by ancestry; those fall back to committer date, and then to
    the hash, so the answer is total and the page does not churn.

    A commit that is unstamped, unknown here, or not on the reported tree cannot be ordered this way
    at all; those fall back to file mtime and then path, and say so.
    """

    def __init__(self, repo: Path):
        self.repo = repo
        self._anc: dict[tuple[str, str], bool] = {}
        self._date: dict[str, int] = {}
        self._on: dict[str, bool] = {}

    def on_tree(self, c: str | None) -> bool:
        if not c:
            return False
        if c not in self._on:
            self._on[c] = git(self.repo, "merge-base", "--is-ancestor", c, "HEAD") == ""
        return self._on[c]

    def _is_ancestor(self, a: str, b: str) -> bool:
        if (a, b) not in self._anc:
            self._anc[(a, b)] = git(self.repo, "merge-base", "--is-ancestor", a, b) == ""
        return self._anc[(a, b)]

    def _when(self, c: str) -> int:
        if c not in self._date:
            v = git(self.repo, "show", "-s", "--format=%ct", c)
            self._date[c] = int(v) if v and v.isdigit() else 0
        return self._date[c]

    def newest(self, obs: list[dict]) -> tuple[dict, list[dict]]:
        """``(current record, superseded records)`` over records of one config and one reference."""
        rankable = [o for o in obs if self.on_tree(o.get("commit"))]
        if not rankable:  # unstamped, or a tree with no git
            ordered = sorted(obs, key=lambda o: (o.get("mtime") or 0, o["path"]))
            return ordered[-1], ordered[:-1]

        def cmp(x, y) -> int:
            a, b = x["commit"], y["commit"]
            if a == b:
                return (x["path"] > y["path"]) - (x["path"] < y["path"])
            if self._is_ancestor(a, b):
                return -1
            if self._is_ancestor(b, a):
                return 1
            return ((self._when(a), a) > (self._when(b), b)) - ((self._when(a), a) < (self._when(b), b))

        import functools

        ordered = sorted(rankable, key=functools.cmp_to_key(cmp))
        return ordered[-1], ordered[:-1] + [o for o in obs if o not in rankable]


# ------------------------------------------------------------------------------ dashboard

#: The dashboard is hand-maintained and has drifted out of valid YAML more than once: an unquoted
#: ``integration_sha: see git log (merges: ...)``, a ``\"`` used as a flow-scalar delimiter, a ``\'``
#: inside a double-quoted scalar. Refusing to run would block the final report on a punctuation bug
#: in a note; repairing silently would mean the source file never gets fixed. So the repairs are
#: minimal, enumerated, and printed on the page.
_REPAIRS = (
    (re.compile(r'(?<=[\[,]\s)\\"'), '"', "a backslash before a flow-scalar opening quote"),
    (re.compile(r'\\"(?=\s*[,\]])'), '"', "a backslash before a flow-scalar closing quote"),
    (re.compile(r"\\'"), "'", r"an invalid \' escape inside a double-quoted scalar"),
)
_KV = re.compile(r"^(\s*(?:- )?[A-Za-z_][\w-]*: )(.+?)\s*$")


def load_dashboard(path: Path) -> tuple[dict, list[str]]:
    """``(dashboard, repairs applied)``. An empty repair list means the file parsed as written."""
    import yaml

    txt = path.read_text()
    try:
        return yaml.safe_load(txt), []
    except Exception:
        pass
    notes: list[str] = []
    for pat, rep, why in _REPAIRS:
        txt, n = pat.subn(rep, txt)
        if n:
            notes.append(f"{n}x {why}")
    lines, quoted = [], 0
    for line in txt.splitlines():
        m = _KV.match(line)
        if m and m.group(2) and m.group(2)[0] not in "\"'[{&*|>#" and ": " in m.group(2):
            lines.append(m.group(1) + "'" + m.group(2).replace("'", "''") + "'")
            quoted += 1
        else:
            lines.append(line)
    if quoted:
        notes.append(f"{quoted}x an unquoted scalar containing a ': '")
    return yaml.safe_load("\n".join(lines)), notes


def as_written(x) -> str:
    """The dashboard value as a reader should see it.

    ``pixels_changed: no`` is a YAML **boolean**, so a naive read prints ``False`` where the file
    says ``no`` — and the page would be quoting a field the coordinator never wrote. Booleans are
    mapped back to the words the schema uses; everything else is passed through untouched.
    """
    if isinstance(x, bool):
        return "no" if x is False else "yes"
    return "" if x is None else str(x)


def first_word(x) -> str:
    """``no (default appearance, all 16 DMC cells)`` -> ``no``.

    The dashboard qualifies its yes/no answers in prose. The assertion is about the answer; the
    qualification is printed beside it rather than thrown away.
    """
    s = as_written(x).strip().lower()
    return re.split(r"[\s(,;]", s, maxsplit=1)[0] if s else ""


# ------------------------------------------------------------------------------ step 3 evaluator


def spec_bar() -> dict:
    """The `ENVIRONMENT_DONE` performance bar, read from the frozen Step 2 spec.

    `{"cpu": "CPU_FAST", "gpu": "GPU_FAST"}` plus the conjuncts each backend certification requires.
    Read rather than restated: Step 2 froze these before any Step 3 number existed, and a report that
    retyped them would be a second copy able to drift from the first.
    """
    d = json.loads((ROOT / SPEC_JSON).read_text())["ladder"]
    return {
        "min": d["environment_done_min_tiers"],
        "requires": d.get("environment_done_requires", []),
        "backend": d.get("backend_certified_requires", {}),
    }


def spec_quote(section: str, kind: str = "bold") -> str | None:
    """The sentence the normative document writes for a section, quoted live rather than paraphrased.

    `kind="bold"` takes the first bolded claim under the heading, `kind="fence"` the first fenced
    block. If the document is reorganised the quote disappears rather than silently going stale, and
    the page says which section it could not find.
    """
    md = ROOT / SPEC_MD
    if not md.is_file():
        return None
    lines = md.read_text(errors="replace").splitlines()
    start = next(
        (i for i, l in enumerate(lines) if l.strip().startswith("#") and l.lstrip("# ").startswith(section + " ")),
        None,
    )
    if start is None:
        return None
    body, fence = [], []
    infence = False
    for l in lines[start + 1 :]:
        if l.strip().startswith("#") and not infence:
            break
        if l.strip().startswith("```"):
            infence = not infence
            continue
        (fence if infence else body).append(l)
    if kind == "fence":
        return " / ".join(x.strip() for x in fence if x.strip()) or None
    m = re.search(r"\*\*(.+?)\*\*", " ".join(body))
    return m.group(1).strip() if m else None


def load_step3():
    """Import ``scripts/regression/semantic/certify.py`` without putting the repo root on ``sys.path``."""
    spec = importlib.util.spec_from_file_location(
        "step3_certify", ROOT / "scripts" / "regression" / "semantic" / "certify.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def step3_representative(labels: list[dict], C) -> dict[tuple[str, str], dict]:
    """One representative Step-3 label per ``(env, device)``, by Step 3's own rule.

    Same rule as ``scripts/regression/semantic/certify.py`` (its ``by_env`` pass): an all-core CPU run is a scaling
    result and may never stand in for a tier that Step 2 certifies at one thread, and among the rest
    the more informative label wins.
    """
    best: dict[tuple[str, str], dict] = {}
    for l in labels:
        if l["device"] == "cpu" and (l.get("threads") or 1) != 1:
            continue
        k = (l["env"], l["device"])
        if k not in best or C._rank_label(l) > C._rank_label(best[k]):
            best[k] = l
    return best


# ------------------------------------------------------------------------------ artifact index


#: A workstream quarantines a shard it no longer stands behind by moving it into a `superseded_*`
#: directory beside the replacement, or by stamping the file itself. Both forms are honoured: a
#: measurement the people who took it have withdrawn must not be the one this page labels a tier
#: from, and it must not simply vanish either — it is listed with the reason, in the appendix.
def superseded_dir(rel: Path) -> bool:
    return any(part.lower() == "superseded" or part.lower().startswith("superseded_") for part in rel.parts)


def readme_reason(path: Path, stop: Path) -> str | None:
    """The first prose line of the nearest README at or above ``path``'s directory.

    The reason a shard was withdrawn is written where the shard was moved to, by the person who
    moved it. Reading it here means the appendix carries their sentence rather than this file's guess
    at why a directory is named what it is named.
    """
    d = path.parent
    while True:
        for r in sorted(d.glob("README*")):
            title = None
            for raw in r.read_text(errors="replace").splitlines():
                line = raw.strip()
                if not line or set(line) <= set("=-_"):
                    continue
                if line.startswith("#"):  # a title, not yet the reason
                    title = title or line.lstrip("#").strip()
                    continue
                return f"{line} (`{r.name}`)"
            if title:  # a README that is only a title
                return f"{title} (`{r.name}`)"
        if d == stop or stop not in d.parents:
            return None
        d = d.parent


def leg_of(rel: Path) -> str | None:
    """``before`` / ``after`` for an artifact, from the DEEPEST path component that says so.

    ``profile_after_2a/before_baseline/X.json`` is a BEFORE leg inside an after-the-patch directory,
    and reading the shallow component would label the pristine baseline as the patched measurement —
    the one mistake that turns a before/after into a fiction.
    """
    for part in reversed(rel.parts):
        if "before" in part:
            return "before"
        if "after" in part:
            return "after"
    return None


class Artifacts:
    """Every JSON under the results tree, classified by shape. Nothing in here may raise."""

    def __init__(self, results: Path):
        self.results = results
        self.unknown: list[tuple[str, str]] = []  # (path, why)
        self.profiles: list[tuple[Path, dict]] = []  # scratch/step6/profile_note.py output
        self.tier_deltas: list[tuple[Path, dict]] = []  # scratch/step6/tilemap/tier.py output
        self.sweeps: list[tuple[Path, dict]] = []  # Step 3 raw renderer benchmark shards
        self.certs: list[tuple[Path, dict]] = []  # Step 4 Gate-A certifications
        self.conformance: list[tuple[Path, dict]] = []  # Step 4 raw conformance shards
        self.pixel: list[tuple[Path, dict]] = []  # before/after pixel comparisons
        self.replay_legs: list[tuple[Path, dict]] = []  # one leg of a Step-4 replay, with a summary
        self.kind: dict[str, str] = {}  # every JSON -> what it was recognised as
        self.superseded: list[tuple[str, str]] = []  # withdrawn by the workstream, with the reason
        self.superseded_other: int = 0
        self.other_files: list[str] = []
        self._census = None
        self._C = None
        self._cert_prov_cache: dict[Path, tuple[str | None, str]] = {}
        self._read()

    # -------------------------------------------------------------- reading

    def rel(self, p: Path) -> str:
        return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)

    def _read(self) -> None:
        for p in sorted(self.results.rglob("*")):
            if not p.is_file():
                continue
            quarantined = superseded_dir(p.relative_to(self.results))
            if p.suffix != ".json":
                if quarantined:
                    if p.name.lower().startswith("readme"):
                        self.superseded.append((self.rel(p), "the note that records the withdrawal"))
                    else:
                        self.superseded_other += 1
                else:
                    self.other_files.append(self.rel(p))
                continue
            try:
                d = json.loads(p.read_text())
            except Exception as e:
                self.unknown.append((self.rel(p), f"does not parse as JSON: {type(e).__name__}"))
                continue
            why = self._withdrawn(p, d, quarantined)
            if why is not None:
                self.superseded.append((self.rel(p), why))
                continue
            try:
                self._classify(p, d)
            except Exception as e:  # a shape that looked known and was not
                self.unknown.append((self.rel(p), f"recognised but could not be read: {type(e).__name__}: {e}"))

    def _withdrawn(self, p: Path, d, quarantined: bool) -> str | None:
        """The reason this artifact was withdrawn, or None if it still counts.

        Three forms, all of them the workstream's own statement and none of them this file's opinion:
        a `superseded_*` directory, a `"superseded": true` stamp, or an `"unusable": "<reason>"`
        string. gfxdraw's node2338 shards and raster2d's round-1 node2344 shards are quarantined this
        way because their host load averages make them a measurement of the node.
        """
        reasons = []
        if quarantined:
            reasons.append("superseded by the workstream")
        if isinstance(d, dict):
            u = d.get("unusable")
            if isinstance(u, str) and u.strip():
                reasons.append(f"marked unusable: {u.strip()}")
            sup = d.get("superseded")
            if isinstance(sup, str) and sup.strip():
                reasons.append(f"marked superseded: {sup.strip()}")
            elif sup is True:
                reasons.append("marked `superseded: true`")
        if not reasons:
            return None
        note = readme_reason(p, self.results)
        if note and quarantined:
            reasons.append(note)
        return "; ".join(reasons)

    def _classify(self, p: Path, d) -> None:
        if p.name == "plan.json":
            self.kind[self.rel(p)] = (
                "the pre-implementation workstream ranking, superseded by the measurements on this page"
            )
            return
        if not isinstance(d, dict):
            self.unknown.append((self.rel(p), f"top level is a {type(d).__name__}, not an object"))
            return
        k = set(d)
        if {"env", "device", "before", "after"} <= k and isinstance(d.get("after"), dict) and "tier" in d["after"]:
            self.tier_deltas.append((p, d))
            self.kind[self.rel(p)] = "a before/after tier artifact"
        elif {"envs", "provenance"} <= k and isinstance(d.get("envs"), list):
            self.sweeps.append((p, d))
            self.kind[self.rel(p)] = "a Step-3 raw renderer benchmark shard"
        elif (
            d.get("step") == 4
            and isinstance(d.get("configs"), dict)
            and any(isinstance(v, dict) and "gate_a" in v for v in d["configs"].values())
        ):
            self.certs.append((p, d))
            self.kind[self.rel(p)] = "a Step-4 Gate-A certification"
        elif d.get("step") == 4 and isinstance(d.get("configs"), dict):
            self.conformance.append((p, d))
            self.kind[self.rel(p)] = (
                "a raw Step-4 conformance shard: per-config measurements with "
                "no verdict applied, which is what the certifications above "
                "are computed from"
            )
        elif {"env", "batches"} <= k and isinstance(d.get("batches"), dict):
            # A profile note and an attribution dump both carry `env` and `batches`. Only the profile
            # note times the public path (`render_ms`), and only it may feed a before/after: letting
            # an attribution dump into that table would divide one kind of millisecond by another.
            cells = [c for c in d["batches"].values() if isinstance(c, dict)]
            if "versions" in k or "tool" in k or any("render_ms" in c for c in cells):
                self.profiles.append((p, d))
                self.kind[self.rel(p)] = "a profile note"
            else:
                self.kind[self.rel(p)] = (
                    "a phase/op attribution dump, not a timing of the public path — not a before/after quantity"
                )
        elif isinstance(d.get("configs"), dict) and isinstance(d.get("summary"), dict) and "pass" in d["summary"]:
            self.replay_legs.append((p, d))
            self.kind[self.rel(p)] = "one leg of a Step-4 replay"
        elif isinstance(d.get("configs"), dict) and any(
            isinstance(v, dict) and "identical" in v for v in d["configs"].values()
        ):
            self.pixel.append((p, d))
            self.kind[self.rel(p)] = "a pixel comparison"
        elif {"config", "digests"} <= k:
            self.pixel.append((p, d))
            self.kind[self.rel(p)] = "a digest comparison"
        else:
            self.unknown.append((self.rel(p), f"unrecognised shape; top-level keys {sorted(k)[:8]}"))

    # -------------------------------------------------------------- derived views

    def workstream_of(self, p: Path) -> str:
        r = p.relative_to(self.results)
        return r.parts[0] if len(r.parts) > 1 else "(top level)"

    def _cert_provenance(self, p: Path, d: dict) -> tuple[str | None, str]:
        """``(git_commit, declared reference version)`` for a Gate-A certification.

        A `certification.json` is a VERDICT computed from a shard, and it is the shard that carries
        the provenance — the commit the measurement ran at and the reference packages it ran against.
        Both are followed here rather than guessed from the directory name, because "which record is
        current" and "which reference arm is this" are the two questions the verdict rule turns on,
        and a directory name is a label somebody typed.
        """
        if p.parent in self._cert_prov_cache:
            return self._cert_prov_cache[p.parent]
        commit = (d.get("provenance") or {}).get("git_commit") if isinstance(d.get("provenance"), dict) else None
        ref: dict = {}
        shards = [d]
        for q in sorted(p.parent.rglob("*.json")):
            if q == p:
                continue
            try:
                shards.append(json.loads(q.read_text()))
            except Exception:
                continue
        for cand in shards:
            if not isinstance(cand, dict):
                continue
            provs = []
            if isinstance(cand.get("provenance"), dict):
                provs.append(cand["provenance"])
            for sh in (cand.get("shards") or {}).values() if isinstance(cand.get("shards"), dict) else []:
                if isinstance(sh, dict) and isinstance(sh.get("provenance"), dict):
                    provs.append(sh["provenance"])
            for prov in provs:
                commit = commit or prov.get("git_commit")
                rp = prov.get("reference_packages")
                if isinstance(rp, dict) and not ref:
                    ref = {k: v for k, v in rp.items() if v}
        label = ", ".join(f"{k} {ref[k]}" for k in sorted(ref) if k in ("mujoco", "dm_control")) or p.parent.name
        self._cert_prov_cache[p.parent] = (commit, label)
        return commit, label

    def gate_a_observations(self) -> dict[str, list[dict]]:
        """``{config: [{status, blocked, path, ws, arm, commit, ref, mtime}]}``."""
        out: dict[str, list[dict]] = defaultdict(list)
        for p, d in self.certs:
            commit, ref = self._cert_provenance(p, d)
            for cfg, v in d["configs"].items():
                if isinstance(v, dict):
                    out[cfg].append(
                        {
                            "status": v.get("gate_a"),
                            "blocked": list(v.get("blocked") or []),
                            "path": self.rel(p),
                            "ws": self.workstream_of(p),
                            "arm": p.parent.name,
                            "commit": commit,
                            "ref": ref,
                            "mtime": p.stat().st_mtime,
                        }
                    )
        return out

    def tier_observations(self) -> dict[tuple[str, str], list[dict]]:
        """``{(env, device): [{tier, qualifiers, leg, host, path, how}]}``.

        Two sources, one meaning. A tier artifact already carries Step 3's label for both legs. A raw
        Step-3 shard carries only cells, so it is put through Step 3's evaluator here — which is a
        re-measurement of the same quantity under the same frozen ladder, and is what "the tier after
        Step 6" has to mean if it is to be comparable with the Step 3 column beside it.
        """
        out: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for p, d in self.tier_deltas:
            dev = "gpu" if d.get("device") in ("cuda", "gpu") else "cpu"
            for leg in ("before", "after"):
                lab = d.get(leg) or {}
                if isinstance(lab, dict) and lab.get("tier"):
                    out[(d["env"], dev)].append(
                        {
                            "tier": lab["tier"],
                            "qualifiers": lab.get("qualifiers") or [],
                            "leg": leg,
                            "path": self.rel(p),
                            "ws": self.workstream_of(p),
                            "host": scrub(lab.get("host") or "?").split(".")[0],
                            "load": None,
                            "how": "recorded by the workstream",
                        }
                    )
        for p, d in self.sweeps:
            prov = d.get("provenance") or {}
            dev_raw = prov.get("device")
            leg = leg_of(p.relative_to(self.results))
            if dev_raw not in ("cpu", "cuda", "gpu") or leg is None:
                continue
            dev = "gpu" if dev_raw in ("cuda", "gpu") else "cpu"
            for e in d["envs"]:
                if not isinstance(e, dict) or not e.get("env"):
                    continue
                try:
                    if self._C is None:
                        self._C = load_step3()
                        self._census = self._C.read_censuses(
                            sorted((ROOT / "tests" / "fixtures" / "regression" / "census").glob("census_*.json"))
                        )
                    lab = self._C.classify(dict(e, _provenance=prov), dev_raw, self._census)
                except Exception as ex:
                    self.unknown.append(
                        (
                            self.rel(p),
                            f"a Step-3 benchmark shard whose {e.get('env')} row could not be "
                            f"labelled: {type(ex).__name__}: {ex}",
                        )
                    )
                    continue
                out[(e["env"], dev)].append(
                    {
                        "tier": lab.get("tier"),
                        "qualifiers": lab.get("qualifiers") or [],
                        "leg": leg,
                        "path": self.rel(p),
                        "ws": self.workstream_of(p),
                        "host": scrub(prov.get("host") or "?").split(".")[0],
                        "load": prov.get("loadavg_end", prov.get("loadavg_start")),
                        "how": "labelled here by `scripts/regression/semantic/certify.py`",
                    }
                )
        return out

    #: A digest comparison whose status is one of these did not move — but `unavailable` did not
    #: COMPARE either, and counting it as a difference is how a page invents a pixel change out of a
    #: missing official digest. The two are reported in separate columns.
    SAME = ("identical",)
    NOT_COMPARED = ("unavailable", "absent", "not_applicable", "skipped")

    @staticmethod
    def _tail(x, n: int = 2) -> str:
        """The last ``n`` components of a path-like field. The artifacts record absolute cluster
        paths here; the tail is what carries the meaning (`merged/before`) and is safe to print."""
        return "/".join(str(x).rstrip("/").split("/")[-n:]) if x else "?"

    def pixel_movements(self) -> list[dict]:
        """Every recorded pixel comparison, reduced to (what was compared, what moved, what was not
        compared at all)."""
        rows = []
        for p, d in self.pixel:
            if isinstance(d.get("configs"), dict):
                moved = sorted(
                    c for c, v in d["configs"].items() if isinstance(v, dict) and v.get("identical") is False
                )
                rows.append(
                    {
                        "path": self.rel(p),
                        "ws": self.workstream_of(p),
                        "n": len(d["configs"]),
                        "moved": moved,
                        "uncompared": 0,
                        "what": f"{self._tail(d.get('before_dir', 'before'))} vs "
                        f"{self._tail(d.get('after_dir', 'after'))}",
                    }
                )
            else:
                st = [x.get("status") for x in (d.get("digests") or []) if isinstance(x, dict)]
                moved = [s for s in st if s not in self.SAME and s not in self.NOT_COMPARED]
                rows.append(
                    {
                        "path": self.rel(p),
                        "ws": self.workstream_of(p),
                        "n": 1,
                        "moved": [d.get("config")] if (moved or d.get("changed")) else [],
                        "uncompared": sum(s in self.NOT_COMPARED for s in st),
                        "what": f"{d.get('reference_kind', 'a replay')} "
                        f"({self._tail(d.get('compared_against', '?'), 1)}), "
                        f"{len(st)} digest(s)",
                    }
                )
        return rows

    def replay_identity(self) -> dict[str, dict]:
        """`{config: {identical, moved, sources}}` over every committed pixel/digest comparison.

        Correctness accounting is owed for all 49 cells whatever their role, and "did the frames move"
        is the part of it that no dashboard sentence can summarise per configuration. Counted here
        from the comparisons themselves so the per-config column is evidence rather than a claim.
        """
        out: dict[str, dict] = defaultdict(lambda: {"identical": 0, "moved": 0, "sources": []})
        for p, d in self.pixel:
            if isinstance(d.get("configs"), dict):
                for cfg, v in d["configs"].items():
                    if not isinstance(v, dict) or v.get("identical") is None:
                        continue
                    out[cfg]["identical" if v["identical"] else "moved"] += 1
                    out[cfg]["sources"].append(self.rel(p))
            elif d.get("config"):
                st = [x.get("status") for x in (d.get("digests") or []) if isinstance(x, dict)]
                cmp_ = [x for x in st if x not in self.NOT_COMPARED]
                if not cmp_:
                    continue
                moved = any(x not in self.SAME for x in cmp_) or bool(d.get("changed"))
                out[d["config"]]["moved" if moved else "identical"] += 1
                out[d["config"]]["sources"].append(self.rel(p))
        return out

    def wall_times(self) -> dict[str, list[dict]]:
        """``{workstream: [row]}`` — render wall time before vs after, paired on the SAME host.

        AGENT_COMMON's same-card rule applied to the page: a before on one node and an after on
        another is not a before/after, so an unpaired leg is reported as unpaired rather than divided
        into a speed-up. Where a workstream measured the same cell in several rounds, the
        lexicographically last directory wins and both files are cited.
        """
        legs: dict[tuple[str, str, str, str, str], tuple[str, Path, dict]] = {}
        for p, d in self.profiles:
            rel = p.relative_to(self.results)
            leg = leg_of(rel)
            if leg is None or not d.get("env"):
                continue
            key = (
                self.workstream_of(p),
                d["env"],
                str(d.get("device")),
                scrub(d.get("host") or "?").split(".")[0],
                leg,
            )
            group = str(rel.parent)
            if key not in legs or group > legs[key][0]:
                legs[key] = (group, p, d)

        def ms(d: dict, b: str | None) -> float | None:
            if b is None:
                return None
            v = (((d.get("batches") or {}).get(b) or {}).get("render_ms") or {}).get("ms_median")
            return float(v) if isinstance(v, (int, float)) else None

        out: dict[str, list[dict]] = defaultdict(list)
        for (ws, env, dev, host, leg), (_, p, d) in sorted(legs.items()):
            if leg == "after":
                if (ws, env, dev, host, "before") in legs:
                    continue  # reported on the before row, paired
                b = max((k for k in (d.get("batches") or {}) if ms(d, k) is not None), key=int, default=None)
                out[ws].append(
                    {
                        "env": env,
                        "device": dev,
                        "host": host,
                        "batch": b,
                        "before": None,
                        "after": ms(d, b),
                        "x": None,
                        "cite": [self.rel(p)],
                    }
                )
                continue
            after = legs.get((ws, env, dev, host, "after"))
            if after is None:
                b = max((k for k in (d.get("batches") or {}) if ms(d, k) is not None), key=int, default=None)
                out[ws].append(
                    {
                        "env": env,
                        "device": dev,
                        "host": host,
                        "batch": b,
                        "before": ms(d, b),
                        "after": None,
                        "x": None,
                        "cite": [self.rel(p)],
                    }
                )
                continue
            da = after[2]
            b = max(
                (k for k in (d.get("batches") or {}) if ms(d, k) is not None and ms(da, k) is not None),
                key=int,
                default=None,
            )
            if b is None:
                continue
            mb, ma = ms(d, b), ms(da, b)
            out[ws].append(
                {
                    "env": env,
                    "device": dev,
                    "host": host,
                    "batch": b,
                    "before": mb,
                    "after": ma,
                    "x": (mb / ma) if ma else None,
                    "cite": [self.rel(p), self.rel(after[1])],
                }
            )
        return out

    def replay_summaries(self) -> dict[str, list[dict]]:
        """`{workstream: [leg]}` — each committed Step-4 replay leg and how many configs it passed."""
        out: dict[str, list[dict]] = defaultdict(list)
        for p, d in self.replay_legs:
            sm = d["summary"]
            out[self.workstream_of(p)].append(
                {
                    "leg": p.stem,
                    "device": d.get("device", "?"),
                    "host": scrub(d.get("host") or "?").split(".")[0],
                    "n": len(d["configs"]),
                    "pass": sm.get("pass"),
                    "fail": sm.get("fail"),
                    "error": sm.get("error"),
                    "path": self.rel(p),
                }
            )
        return out

    def slurm_jobs(self) -> dict[str, set[str]]:
        """Job ids the artifacts themselves record, per workstream."""
        out: dict[str, set[str]] = defaultdict(set)
        for group in (
            self.profiles,
            self.tier_deltas,
            self.sweeps,
            self.certs,
            self.conformance,
            self.pixel,
            self.replay_legs,
        ):
            for p, d in group:
                found = [d.get("slurm_job"), d.get("slurm_job_id")]
                prov = d.get("provenance")
                if isinstance(prov, dict):
                    found.append(prov.get("slurm_job"))
                for v in found:
                    if v not in (None, "", "None"):
                        out[self.workstream_of(p)].add(str(v))
        return out


def compare_txt(results: Path) -> dict[str, dict]:
    """``*/replay/compare.txt`` — the only Step-6 artifact that records ``declaration_changed``, and
    it records it as a line of text. Parsed rather than read by eye, like everything else here."""
    rows: dict[str, dict] = {}
    pat = re.compile(r"^(\w+)\s+(pixel-identical|PIXELS MOVED)\s+declaration_changed=(True|False)")
    for p in sorted(results.rglob("compare.txt")):
        for line in p.read_text().splitlines():
            m = pat.match(line)
            if m:
                rows[m.group(1)] = {
                    "config": m.group(1),
                    "pixels": m.group(2),
                    "declaration_changed": m.group(3) == "True",
                    "path": str(p.relative_to(ROOT)),
                    "ws": p.relative_to(results).parts[0],
                }
    return rows


# ------------------------------------------------------------------------------ markdown


def esc(x) -> str:
    return scrub("—" if x is None or x == "" else x).replace("|", "\\|").replace("\n", " ")


def tbl(rows, head) -> list[str]:
    if not rows:
        return ["_none._", ""]
    r = [[esc(c) for c in row] for row in rows]
    w = [max(len(esc(h)), *(len(x[i]) for x in r)) for i, h in enumerate(head)]
    out = [
        "| " + " | ".join(esc(h).ljust(w[i]) for i, h in enumerate(head)) + " |",
        "|" + "|".join("-" * (x + 2) for x in w) + "|",
    ]
    out += ["| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(row)) + " |" for row in r]
    return out + [""]


def num(x, n=2) -> str:
    return "—" if x is None else f"{x:,.{n}f}"


class Sources:
    """Footnote keys for cited files, so a 49-row table can cite every cell and stay readable."""

    def __init__(self):
        self.keys: dict[str, str] = {}

    def key(self, path: str | None) -> str:
        if not path:
            return "—"
        if path not in self.keys:
            self.keys[path] = f"s{len(self.keys) + 1}"
        return self.keys[path]

    def legend(self) -> list[list[str]]:
        return [[f"`{k}`", f"`{p}`"] for p, k in self.keys.items()]


#: Sentence split that will not cut "1.75x" or "e.g. the": only a period or semicolon followed by
#: whitespace and something that starts a new clause.
_SENT = re.compile(r"(?<=[.;])\s+(?=[A-Z`(\[])")


def sentences(text) -> list[str]:
    return [s.strip() for s in _SENT.split(str(text)) if s.strip()]


# ------------------------------------------------------------------------------ the page

#: A sentence in the dashboard that contains one of these is a decision that has not been taken.
#: Keyword extraction, not judgement: the sentence is reprinted verbatim so the words stay the
#: coordinator's.
#: A sentence naming work that was identified and not done inside Step 6. Used only when the
#: dashboard has no `deferred_work` entries, so that an unpopulated field cannot read as an empty
#: backlog. Keyword extraction, verbatim output: the sentence stays the workstream's.
DEFERRED_WORDS = (
    "step-8",
    "step 8",
    "deferred",
    "residual",
    "out of scope",
    "not funded",
    "next target",
    "follow-on",
    "carry-over",
    "carry over",
    "do not build",
    "remains possible",
    "unfunded",
)


def candidate_deferred(ws_list) -> list[list[str]]:
    rows = []
    for w in ws_list:
        blobs = [("notes", t) for t in (w.get("notes") or [])]
        if (w.get("review") or {}).get("verdict"):
            blobs.append(("review", w["review"]["verdict"]))
        for field, blob in blobs:
            for sent in sentences(blob):
                if any(k in sent.lower() for k in DEFERRED_WORDS):
                    rows.append([f"`{w['workstream']}`", f"`{field}`", sent])
    return rows


ESCALATION_WORDS = (
    "escalat",
    "sami",
    "held",
    "pending",
    "open question",
    "not authorized",
    "sign-off",
    "decide",
    "decision",
    "approval",
    "blocker",
    "unresolved",
)

#: A tier name written into the dashboard's free-text performance fields, so the prose can be checked
#: against the artifacts without anybody reading both.
_TIER_TOKEN = re.compile(r"\b(?:CPU|GPU)_[A-Z]+\b")


def build(results: Path, repo: Path, args) -> str:
    dash, repairs = load_dashboard(results / "dashboard.yaml")
    ws_list = list(dash.get("workstreams") or [])
    A = Artifacts(results)
    src = Sources()
    flags: list[list[str]] = []

    def flag(where: str, dash_says, art_says, path: str) -> None:
        flags.append([where, esc(dash_says), esc(art_says), f"`{path}`"])

    S4 = "tests/fixtures/regression/join/step4/certification.json"
    S3 = "tests/fixtures/regression/join/step3/certification.json"
    step4 = json.loads((ROOT / S4).read_text())["configs"]
    step3 = json.loads((ROOT / S3).read_text())["labels"]
    rep3 = step3_representative(step3, load_step3())

    # ---------------------------------------------------------------- SHAs
    head = args.integration_sha or git(repo, "rev-parse", "HEAD")
    branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    base = args.baseline_sha or dash.get("baseline_sha")
    base_how = "`--baseline-sha`" if args.baseline_sha else "`dashboard.yaml: baseline_sha`"
    # The page names the tree it was generated FROM, which is necessarily the parent of the commit
    # that carries the page. Saying so is cheaper than the alternative, which is a reader wondering
    # why the SHA on the page is one behind the branch.
    head_how = (
        "`--integration-sha`"
        if args.integration_sha
        else (
            f"`git rev-parse HEAD` in this tree (`{branch}`) when the page was generated; the "
            f"commit that adds the page is its child"
            if head
            else "unavailable: this tree is not a git checkout"
        )
    )
    anc = git(repo, "merge-base", "--is-ancestor", str(base), "HEAD")
    ancestry = (
        "the baseline is an ancestor of this tree"
        if anc == ""
        else "NOT verified here: `git merge-base --is-ancestor` did not succeed"
    )
    merges = merge_commits(repo) if head and not args.integration_sha else merge_commits(repo)

    L: list[str] = []
    P = L.append
    P("# Step 6 — optimising and repairing the existing renderer paths")
    P("")
    P(
        "Generated by `scripts/regression/optimization/report.py` from `tests/fixtures/regression/join/step6/`, from the frozen Step 3"
    )
    P("labels and from the frozen Step 4 certification. Every number and every count on this page is")
    P("computed from one of those files and names it; nothing here is maintained by hand. Where the")
    P("coordinator's dashboard and a committed artifact disagree the page prints the **artifact**, and")
    P("the disagreement is listed under *Where the dashboard and the artifacts disagree*.")
    P("")

    # "Merged" is decided by git, not by the status string. `raster2d` reports itself *partially*
    # merged and `raycast3d` reports itself still in implementation, yet both have code on this
    # branch — and code on the branch is exactly what the Step-5 semantic assertion below has to
    # cover. A status prefix would have excluded them and made the assertion narrower than the tree.
    says_merged = [w for w in ws_list if str(w.get("status", "")).lower().startswith("merged")]
    merged = [w for w in ws_list if merges.get(w["workstream"])]
    merged_basis = (
        "**a merge commit on this branch** — not by a status string, because "
        "`raster2d` calls itself partially merged and `raycast3d` calls itself still "
        "in implementation while the code of both is in the tree this page describes"
    )
    if not merges:  # no git here: an Oscar copy, or a tarball
        merged, merged_basis = (
            says_merged,
            (
                "the dashboard's `status` field, because no git log is readable from this tree — a "
                "weaker basis, since a partially merged workstream does not say `merged` and is "
                "therefore missing from the set below"
            ),
        )

    P("## Baseline, integration, and where Step 6 stands")
    P("")
    L.extend(
        tbl(
            [
                ["baseline SHA", f"`{base}`", base_how],
                ["integration SHA", f"`{head}`" if head else "—", head_how],
                ["integration branch", f"`{dash.get('integration_branch')}`", "`dashboard.yaml`"],
                ["baseline ancestry", ancestry, "`git merge-base --is-ancestor`"],
                ["workstreams", str(len(ws_list)), "`dashboard.yaml: workstreams`"],
                ["workstreams with code on this branch", f"{len(merged)} of {len(ws_list)}", "`git log --merges`"],
                [
                    "workstreams whose status says merged",
                    f"{len(says_merged)} of {len(ws_list)}",
                    "`dashboard.yaml: status`",
                ],
                ["merge commits on this branch", str(sum(len(v) for v in merges.values())), "`git log --merges`"],
            ]
            + (
                [["Step 6 status", dash["step6_status"], "`dashboard.yaml: step6_status`"]]
                if dash.get("step6_status")
                else []
            ),
            ["", "value", "read from"],
        )
    )
    iv = dash.get("integration_verification")
    if isinstance(iv, dict):
        P("The integration branch was verified as a whole, separately from the per-workstream")
        P("evidence below. That record, in full:")
        P("")
        L.extend(
            tbl(
                [[f"`{k}`", v if not isinstance(v, list) else ", ".join(str(x) for x in v)] for k, v in iv.items()],
                ["field", "`dashboard.yaml: integration_verification`"],
            )
        )

    if repairs:
        P('!!! warning "`dashboard.yaml` does not parse as YAML and was repaired in order to read it"')
        P("    The repairs are textual and minimal, and they are listed here so that the source file")
        P("    gets fixed rather than the workaround becoming permanent: " + "; ".join(repairs) + ".")
        P("")

    # ---------------------------------------------------------------- workstream table
    P("## Workstreams")
    P("")
    P("One row per workstream, in the fields section 19 defines. `merge commit` is read from")
    P("`git log --merges` on this branch rather than from the dashboard, so a workstream that reports")
    P("itself merged and has no merge commit here becomes a flagged disagreement rather than a claim.")
    P("")
    rows = []
    for w in ws_list:
        name = w["workstream"]
        mc = merges.get(name, [])
        rows.append(
            [
                f"`{name}`",
                str(w.get("status", "")).split(" (")[0],
                str(w.get("agent_model")),
                str(w.get("authorization", "")),
                str(len(w.get("affected_configs") or [])),
                first_word(w.get("pixels_changed")),
                first_word(w.get("semantic_followup")),
                (w.get("review") or {}).get("reviewer") or "—",
                ", ".join(f"`{s[:7]}`" for s, _ in mc) or "—",
            ]
        )
        if str(w.get("status", "")).lower().startswith("merged") and not mc:
            flag(
                f"`{name}`: merge commit",
                w.get("status"),
                f"no `Merge step6/{name}` commit is reachable from this branch",
                "git log --merges",
            )
    L.extend(
        tbl(
            rows,
            [
                "workstream",
                "status",
                "model",
                "authorization",
                "configs",
                "pixels",
                "semantic",
                "reviewer",
                "merge commit",
            ],
        )
    )

    # ---------------------------------------------------------------- the 49 configs
    gate_obs = A.gate_a_observations()
    tier_obs = A.tier_observations()
    decl = compare_txt(results)

    bar = spec_bar()
    C3 = load_step3()
    rank = C3._tier_rank
    ident = A.replay_identity()

    def missing(t) -> bool:
        """No measurement at all, as distinct from a measurement that came out low."""
        return not t or t.endswith("UNMEASURED")

    def meets(t, dev: str) -> bool:
        return bool(t) and rank(t) >= rank(bar["min"][dev])

    P("## Scope: which cells owe correctness, and which owe throughput")
    P("")
    P("These are not the same set, and Step 6 spent a week reporting them as if they were.")
    P("")
    q42, q122 = spec_quote("4.2"), spec_quote("12.2", "fence")
    P(
        f"Step 2 §4.2 — `{SPEC_MD}`: "
        + (
            f"**{q42}**"
            if q42
            else "_(the section could not be located in the document; the rule below is applied from the "
            "machine-readable spec alone)_"
        )
    )
    P("")
    P(f"Step 2 §12.2 puts completion on that same default — `{SPEC_MD}`:")
    P("")
    if q122:
        P("```text")
        for line in q122.split(" / "):
            P(line)
        P("```")
    else:
        P("_(the section could not be located in the document)_")
    P("")
    P("So the accounting splits, and the page splits with it:")
    P("")
    L.extend(
        tbl(
            [
                [
                    "correctness and fidelity — Gate A, replay identity, digests",
                    f"all {len(step4)} render-distinct cells",
                    "a parameter-space cell exists to prove the renderer answers the whole parameter space; "
                    "getting it wrong is a defect wherever it happens",
                ],
                [
                    "performance completion — CPU_FAST+, GPU_FAST+, remaining blockers",
                    f"the {sum(1 for c in step4.values() if c.get('role') == OWES_PERFORMANCE)} "
                    f"`{OWES_PERFORMANCE}` cells only",
                    "§4.2 and §12.2 put the certified configuration and `ENVIRONMENT_DONE` on the registered "
                    "SWM default; no other cell is owed a tier",
                ],
            ],
            ["accounting", "denominator", "why that denominator"],
        )
    )
    P(
        "The bar itself is read from the frozen spec, not restated here — "
        f"`{SPEC_JSON}`, `ladder.environment_done_min_tiers`: "
        + ", ".join(f"**{d.upper()} >= `{t}`**" for d, t in sorted(bar["min"].items()))
        + "."
    )
    P("")

    # ------------------------------------------------------------ correctness, all 49
    P(f"## Correctness after Step 6 — all {len(step4)} frozen configurations")
    P("")
    P("- **Gate A** is the Step 4 status, *overridden only where a Step 6 certification artifact")
    P("  states a status for that configuration*. Where two Step 6 arms disagree — the")
    P("  texture-fidelity 3.11 and 3.12 reference pairs are two arms of one diagnosis — the arm that")
    P("  ran under the declared reference, the one whose certifier blocked nothing, is the status;")
    P("  both arms are printed in the detail column and the disagreement is never hidden.")
    P("- **replay identity** counts the committed before/after pixel and digest comparisons that")
    P("  mention the cell. `identical xN` means N comparisons found no moved pixel; a cell with no")
    P("  comparison says so rather than being credited with one.")
    P("")

    table, gate_ok = [], 0
    gate_of, tier_of, ident_of = {}, {}, {}
    order = CommitOrder(repo)
    superseded_certs: list[list[str]] = []
    for cfg in sorted(step4):
        c4 = step4[cfg]
        g4 = c4.get("gate_a")
        allobs = sorted(gate_obs.get(cfg, []), key=lambda o: o["path"])
        detail: list[str] = []
        # Two records of the same config under the SAME declared reference are two runs of one
        # measurement, and only the newest is the verdict. Grouping by reference first is what keeps
        # that from silently collapsing the 3.11 and 3.12 arms, which are two different questions.
        by_ref: dict[str, list[dict]] = defaultdict(list)
        for o in allobs:
            by_ref[o["ref"]].append(o)
        obs = []
        for ref in sorted(by_ref):
            cur, old_ones = order.newest(by_ref[ref])
            obs.append(cur)
            for o in old_ones:
                superseded_certs.append(
                    [
                        f"`{cfg}`",
                        o["ref"],
                        o["arm"],
                        o["status"] + (" (blocked)" if o["blocked"] else ""),
                        f"`{(o['commit'] or 'unstamped')[:12]}`",
                        f"`{(cur['commit'] or 'unstamped')[:12]}` ({cur['arm']})",
                        f"`{o['path']}`",
                    ]
                )
        if obs:
            clean = [o for o in obs if not o["blocked"]]
            chosen = (clean or obs)[0]
            gate, gate_src = chosen["status"], src.key(chosen["path"])
            for o in obs:
                detail.append(
                    f"{o['ref']} @ `{(o['commit'] or 'unstamped')[:8]}`: {o['status']}"
                    f"{' (blocked)' if o['blocked'] else ''} [{src.key(o['path'])}]"
                )
            if len({o["status"] for o in obs}) > 1:
                detail.append("**arms disagree**")
            if g4 != gate:
                detail.append(f"Step 4 said {g4}")
        else:
            gate, gate_src = g4, src.key(S4)
        d = decl.get(cfg)
        if d and d["declaration_changed"]:
            detail.append(f"declaration changed [{src.key(d['path'])}]")
        if d and d["pixels"] == "PIXELS MOVED":
            detail.append(f"replay pixels moved against the frozen digests [{src.key(d['path'])}]")

        cells = {}
        for dev in ("cpu", "gpu"):
            after = sorted((o for o in tier_obs.get((cfg, dev), []) if o["leg"] == "after"), key=lambda o: o["path"])
            if after:
                cells[dev] = (after[-1]["tier"], src.key(after[-1]["path"]), len({o["tier"] for o in after}) > 1)
            else:
                r = rep3.get((cfg, dev))
                cells[dev] = ((r or {}).get("tier"), src.key(S3) if r else None, False)
        if cells["cpu"][2]:
            detail.append("**CPU after-legs disagree**")
        if cells["gpu"][2]:
            detail.append("**GPU after-legs disagree**")

        ri = ident.get(cfg)
        if ri and not ri["moved"]:
            id_txt, id_src = f"identical x{ri['identical']}", src.key(sorted(ri["sources"])[0])
        elif ri:
            id_txt, id_src = (
                f"**moved in {ri['moved']} of {ri['moved'] + ri['identical']}**",
                src.key(sorted(ri["sources"])[0]),
            )
        else:
            id_txt, id_src = "no committed comparison", None

        gate_ok += gate == "PASS"
        gate_of[cfg], tier_of[cfg], ident_of[cfg] = (gate, gate_src), cells, (id_txt, id_src)
        table.append(
            [f"`{cfg}`", c4.get("role", ""), g4, gate, gate_src, id_txt, id_src or "—", "; ".join(detail) or "—"]
        )
    L.extend(
        tbl(
            table,
            ["config", "role", "Gate A (Step 4)", "Gate A (after Step 6)", "src", "replay identity", "src", "detail"],
        )
    )
    P("### Superseded certification records")
    P("")
    P("A workstream that re-runs a certification commits the new record beside the old one. Both are")
    P("evidence; only one is the verdict. Among records for the same configuration **and the same")
    P("declared reference version**, the current record is the one whose stamped `git_commit` is")
    P("newest by ancestry on the tree reported above (`git merge-base --is-ancestor`; committer date")
    P("then hash where two commits are on branches that never saw each other; file mtime only where")
    P("a record is unstamped). The rest are listed here and were used for nothing:")
    P("")
    L.extend(
        tbl(
            superseded_certs,
            ["config", "declared reference", "record", "it said", "at commit", "superseded by", "artifact"],
        )
    )

    # ------------------------------------------------------------ performance, registered defaults
    owed = sorted(c for c in step4 if step4[c].get("role") == OWES_PERFORMANCE)
    other = sorted(c for c in step4 if step4[c].get("role") != OWES_PERFORMANCE)
    P(f"## Performance after Step 6 — the {len(owed)} registered/default configurations")
    P("")
    P("The cells §4.2 and §12.2 put the contract on. **CPU tier / GPU tier** are the representative")
    P("Step 3 labels, *overridden only where a Step 6 artifact re-measures that row*; a re-measured")
    P("tier is read from a tier artifact or computed from the raw shard by `scripts/regression/semantic/certify.py`,")
    P("Step 3's own evaluator against Step 2's frozen ladder, never a second implementation of it.")
    P("`missing` is an absence of evidence — no measurement exists — and is counted apart from a")
    P("measurement that came out below the bar.")
    P("")
    perf, cpu_ok, gpu_ok, cpu_missing, gpu_below, blockers = [], 0, 0, 0, 0, []
    dist: dict[str, dict[str, int]] = {"cpu": defaultdict(int), "gpu": defaultdict(int)}
    remeasured = {"cpu": 0, "gpu": 0}
    for cfg in owed:
        (ct, cs, _), (gt, gs, _) = tier_of[cfg]["cpu"], tier_of[cfg]["gpu"]
        gate = gate_of[cfg][0]
        cpu_ok += meets(ct, "cpu")
        gpu_ok += meets(gt, "gpu")
        cpu_missing += missing(ct)
        gpu_below += not meets(gt, "gpu")
        dist["cpu"][ct or "never benchmarked"] += 1
        dist["gpu"][gt or "never benchmarked"] += 1
        remeasured["cpu"] += bool(cs and cs != src.keys.get(S3))
        remeasured["gpu"] += bool(gs and gs != src.keys.get(S3))
        why = []
        if gate != "PASS":
            why.append(f"Gate A {gate}")
        if missing(ct):
            why.append(f"CPU missing ({ct or 'never benchmarked'})")
        elif not meets(ct, "cpu"):
            why.append(f"CPU {ct} below {bar['min']['cpu']}")
        if missing(gt):
            why.append(f"GPU missing ({gt or 'never benchmarked'})")
        elif not meets(gt, "gpu"):
            why.append(f"GPU {gt} below {bar['min']['gpu']}")
        if why:
            blockers.append(
                [
                    f"`{cfg}`",
                    "; ".join(why),
                    ", ".join(f"{lbl} {k}" for lbl, k in (("CPU", cs), ("GPU", gs), ("Gate A", gate_of[cfg][1])) if k),
                ]
            )
        perf.append(
            [
                f"`{cfg}`",
                ct or "never benchmarked",
                cs or "—",
                gt or "never benchmarked",
                gs or "—",
                "yes" if not why else "no",
            ]
        )
    L.extend(tbl(perf, ["config", "CPU tier", "src", "GPU tier", "src", "meets the §12.2 bar"]))

    # ------------------------------------------------------------ parameter space
    P(f"## Render-parameter-space configurations — the other {len(other)}")
    P("")
    P("These cells are certified for **correctness**: they exist to show that the renderer answers")
    P("the whole declared parameter space, not only the registered default. The frozen contract asks")
    P("them for no tier, so a tier measured here is **reported, not owed**, and none of them is ever")
    P("counted as a performance blocker. Reporting them as blockers — which this page did until")
    P("Sami's 2026-09-15 decision — inflates the remaining work with an obligation nobody wrote down.")
    P("")
    L.extend(
        tbl(
            [
                [
                    f"`{cfg}`",
                    step4[cfg].get("role", ""),
                    gate_of[cfg][0],
                    gate_of[cfg][1],
                    ident_of[cfg][0],
                    ident_of[cfg][1] or "—",
                    tier_of[cfg]["cpu"][0] or "not measured",
                    tier_of[cfg]["gpu"][0] or "not measured",
                    "not required by the frozen contract",
                ]
                for cfg in other
            ],
            [
                "config",
                "role",
                "Gate A (after Step 6)",
                "src",
                "replay identity",
                "src",
                "CPU tier (reported)",
                "GPU tier (reported)",
                "performance",
            ],
        )
    )

    P("Every `src` key on this page:")
    P("")
    L.extend(tbl(src.legend(), ["key", "file"]))

    # ------------------------------------------------------------ the summary
    n, m = len(step4), len(owed)
    P("### The section-25 summary")
    P("")
    L.extend(
        tbl(
            [
                ["Gate-A pass", f"{gate_ok} / {n}", f"all {n} render-distinct cells"],
                ["registered/default CPU_FAST+", f"{cpu_ok} / {m}", f"tier >= `{bar['min']['cpu']}`"],
                ["registered/default GPU_FAST+", f"{gpu_ok} / {m}", f"tier >= `{bar['min']['gpu']}`"],
                [
                    "registered/default rows with CPU missing",
                    f"{cpu_missing} / {m}",
                    "no CPU measurement exists for the row",
                ],
                [
                    "registered/default rows below the GPU target",
                    f"{gpu_below} / {m}",
                    f"does not reach `{bar['min']['gpu']}` (a missing measurement included, and named as such "
                    f"in the row below)",
                ],
                ["remaining registered/default blockers", f"{len(blockers)} / {m}", "fails any conjunct of §12.2"],
            ],
            ["count", "value", "what it means"],
        )
    )
    P("Each remaining blocker, with the conjunct that blocks it and the artifact each cell was read")
    P("from:")
    P("")
    # The whole-branch verification counted Gate A too, on its own run. Two independent counts of the
    # same quantity is a gift, and printing both without comparing them would waste it.
    iv_gate = str(((dash.get("integration_verification") or {}).get("gate_a")) or "")
    m = re.search(r"(\d+)\s*PASS", iv_gate)
    if m and int(m.group(1)) != gate_ok:
        flag(
            "Gate-A pass count",
            f"`integration_verification.gate_a`: {iv_gate}",
            f"{gate_ok} of {n} computed from the Step-4 certification and the Step-6 certification "
            f"artifacts on this branch",
            "tests/fixtures/regression/join/step6/dashboard.yaml",
        )
    L.extend(tbl(blockers, ["config", "blocking conjunct", "src"]))
    P("A blocker on this list is a cell that has not reached the bar. It is **not** a statement that")
    P("nothing further can be done for it: the work that was identified and then not funded is listed")
    P("under *Deferred possible work*, and the cases where an attempt was made and the ceiling was")
    P("measured are under *Demonstrated blockers*. Reading this table without those two sections")
    P('turns "not done" into "impossible", which is the one conclusion the evidence does not')
    P("support.")
    P("")
    P(f"Counted by tier over the {m} registered/default cells. {remeasured['cpu']} CPU cell(s) and")
    P(f"{remeasured['gpu']} GPU cell(s) were re-measured by a Step 6 artifact; the rest still carry")
    P("their Step 3 label:")
    P("")
    # The tie-break on the name is not decoration: without it the order of two equally-populated
    # tiers comes out of set iteration, and the page churns between runs that measured nothing new.
    order = sorted(
        {t for d in dist.values() for t in d}, key=lambda t: (t.split("_")[0], -dist["cpu"][t] - dist["gpu"][t], t)
    )
    L.extend(
        tbl(
            [[t, str(dist["cpu"].get(t, 0)), str(dist["gpu"].get(t, 0))] for t in order],
            ["tier after Step 6", "CPU configs", "GPU configs"],
        )
    )

    # ---------------------------------------------------------------- per workstream
    wave0 = {str(x) for x in (dash.get("wave0_profile_jobs") or [])}
    walls = A.wall_times()
    replays = A.replay_summaries()
    jobs = A.slurm_jobs()
    pix = A.pixel_movements()
    P("## Per workstream")
    P("")
    for w in ws_list:
        name = w["workstream"]
        corr, perf = w.get("correctness") or {}, w.get("performance") or {}
        bn = w.get("bottleneck") or {}
        mc = merges.get(name, [])
        P(f"### `{name}`")
        P("")
        L.extend(
            tbl(
                [
                    ["brief", f"`{w.get('brief')}`"],
                    ["branch", f"`{w.get('branch')}`"],
                    ["status", w.get("status")],
                    ["authorization", w.get("authorization")],
                    ["goal", w.get("goal")],
                    [
                        "affected configurations",
                        f"{len(w.get('affected_configs') or [])}: "
                        + ", ".join(f"`{c}`" for c in (w.get("affected_configs") or [])),
                    ],
                    ["bottleneck (measured)", bn.get("measured")],
                    ["bottleneck (hypothesis)", bn.get("hypothesis")],
                    ["correctness — targeted tests", corr.get("targeted_tests")],
                    ["correctness — Step-4 replay", corr.get("step4_replay")],
                    ["correctness — Gate A", corr.get("gate_a")],
                    ["correctness — fidelity", corr.get("fidelity")],
                    ["CPU before", perf.get("cpu_before")],
                    ["CPU after", perf.get("cpu_after")],
                    ["GPU before", perf.get("gpu_before")],
                    ["GPU after", perf.get("gpu_after")],
                    ["BATCH_SCALES", perf.get("batch_scales")],
                    ["NO_LARGE_BATCH_REGRESSION", perf.get("no_large_batch_regression")],
                    ["pixels changed", as_written(w.get("pixels_changed"))],
                    ["semantic follow-up", as_written(w.get("semantic_followup"))],
                    [
                        "review",
                        f"{(w.get('review') or {}).get('reviewer') or '—'} — "
                        f"{(w.get('review') or {}).get('verdict') or 'no verdict recorded'}",
                    ],
                    ["merge commit", ", ".join(f"`{s[:12]}` {subj}" for s, subj in mc) or "not merged"],
                ],
                ["field", "`dashboard.yaml`"],
            )
        )

        rows = walls.get(name) or []
        if rows:
            P("Render wall time from the committed profile notes, paired on the same host, at the")
            P("largest batch both legs measured:")
            P("")
            L.extend(
                tbl(
                    [
                        [
                            f"`{r['env']}`",
                            r["device"],
                            r["host"],
                            r["batch"],
                            num(r["before"], 3),
                            num(r["after"], 3),
                            (num(r["x"]) + "x") if r["x"] else "—",
                            ", ".join(f"`{c}`" for c in r["cite"]),
                        ]
                        for r in rows
                    ],
                    ["config", "device", "host", "B", "before ms", "after ms", "speed-up", "artifact"],
                )
            )
        else:
            P("No committed profile note pairs a before leg with an after leg for this workstream.")
            P("")

        art_tiers = sorted(
            ((k, o) for k, v in tier_obs.items() for o in v if o["ws"] == name and o["leg"] == "after"),
            key=lambda x: (x[0], x[1]["path"]),
        )
        if art_tiers:
            P("Tiers re-measured by this workstream's own committed artifacts:")
            P("")
            P("`load` is the host's load average while the shard was taken, as the shard records it:")
            P("a re-measurement on a loaded shared node is evidence about that node as much as about")
            P("the renderer, and the number belongs beside the tier rather than in a footnote.")
            P("")
            L.extend(
                tbl(
                    [
                        [
                            f"`{k[0]}`",
                            k[1],
                            o["tier"],
                            ", ".join(o["qualifiers"]) or "—",
                            o["host"],
                            num(o.get("load"), 1),
                            o["how"],
                            f"`{o['path']}`",
                        ]
                        for k, o in art_tiers
                    ],
                    ["config", "device", "tier after", "qualifiers", "host", "load", "labelled", "artifact"],
                )
            )
        legs = replays.get(name) or []
        if legs:
            P("Step-4 replay legs this workstream committed, with the pass/fail each artifact records")
            P("for itself. `pass` is measured against the FROZEN oracle, which for most of the corpus")
            P("was taken on CUDA — so a CPU leg failing everywhere is a statement about which device")
            P("the oracle was frozen on, not about the patch. What a patch has to show is that its")
            P("after leg is no worse than its own baseline leg, and that comparison is made below.")
            P("")
            L.extend(
                tbl(
                    [
                        [
                            l["leg"],
                            l["device"],
                            l["host"],
                            str(l["n"]),
                            str(l["pass"]),
                            str(l["fail"]),
                            str(l["error"]),
                            f"`{l['path']}`",
                        ]
                        for l in legs
                    ],
                    ["leg", "device", "host", "configs", "pass", "fail", "error", "artifact"],
                )
            )
            by_leg = {l["leg"]: l for l in legs}
            for l in legs:
                base = None
                for a, b in (
                    ("after", "before"),
                    ("after", "before_baseline"),
                    ("branch", "baseline"),
                    ("_on", "_off"),
                ):
                    cand = l["leg"].replace(a, b)
                    if a in l["leg"] and cand in by_leg:
                        base = by_leg[cand]
                        break
                if base is None:
                    continue
                bad = (l["fail"] or 0) + (l["error"] or 0)
                was = (base["fail"] or 0) + (base["error"] or 0)
                if bad > was:
                    flag(
                        f"`{name}`: replay leg `{l['leg']}`",
                        f"`correctness.step4_replay`: {corr.get('step4_replay')}",
                        f"{bad} fail+error against {was} on its own baseline leg `{base['leg']}`",
                        l["path"],
                    )

        # Two directions, and both are worth having. Forward: a tier an artifact measured that the
        # dashboard's prose never names. Backward: a tier the dashboard claims for which no committed
        # artifact re-measures anything at all — which is the shape a merged-but-unevidenced claim
        # takes, and the shape the raycast3d GPU_FAST claim currently has.
        claimed = {dev: set(_TIER_TOKEN.findall(str(perf.get(f"{dev}_after") or ""))) for dev in ("cpu", "gpu")}
        for k, o in art_tiers:
            dev = "gpu" if k[1] == "gpu" else "cpu"
            if o["tier"] and o["tier"] not in claimed[dev]:
                flag(
                    f"`{name}`: `{k[0]}` {k[1]} tier after Step 6",
                    f"`performance.{dev}_after` mentions "
                    + (", ".join(f"`{t}`" for t in sorted(claimed[dev])) or "no tier name"),
                    f"`{o['tier']}` on {o['host']}",
                    o["path"],
                )
        measured = {"cpu": set(), "gpu": set()}
        for k, o in art_tiers:
            measured["gpu" if k[1] == "gpu" else "cpu"].add(o["tier"])
        for dev in ("cpu", "gpu"):
            missing = sorted(claimed[dev] - measured[dev])
            if missing:
                flag(
                    f"`{name}`: `performance.{dev}_after`",
                    "mentions "
                    + ", ".join(f"`{t}`" for t in missing)
                    + " (a text match on the field, not a reading of the sentence around it)",
                    f"no committed Step-6 artifact re-measures a tier for this workstream on {dev}"
                    if not measured[dev]
                    else "the committed artifacts label " + ", ".join(f"`{t}`" for t in sorted(measured[dev])),
                    f"tests/fixtures/regression/join/step6/{name}/",
                )

        mine = [r for r in pix if r["ws"] == name]
        if mine:
            P("Pixel comparisons this workstream committed:")
            P("")
            L.extend(
                tbl(
                    [
                        [
                            r["what"],
                            str(r["n"]),
                            ", ".join(f"`{c}`" for c in r["moved"]) or "none",
                            str(r["uncompared"]) if r["uncompared"] else "—",
                            f"`{r['path']}`",
                        ]
                        for r in mine
                    ],
                    ["comparison", "configs", "pixels moved", "not compared", "artifact"],
                )
            )
            if first_word(w.get("pixels_changed")) == "no":
                for r in mine:
                    if r["moved"]:
                        flag(
                            f"`{name}`: `pixels_changed`",
                            as_written(w.get("pixels_changed")),
                            f"{len(r['moved'])} configuration(s) are not identical in "
                            f"{r['what']}: " + ", ".join(r["moved"]),
                            r["path"],
                        )

        # The wave-0 profiling jobs produced the `profile_before` notes for several workstreams at
        # once, so the dashboard records them once at the top level rather than eight times. They
        # count as recorded for every workstream; without that they are eight false disagreements.
        recorded = {str(x) for x in (w.get("slurm_jobs") or [])} | wave0
        extra = sorted(j for j in jobs.get(name, set()) if j.isdigit() and j not in recorded)
        if extra:
            flag(
                f"`{name}`: `slurm_jobs`",
                f"{len(recorded)} job(s) recorded for this workstream (its own plus the "
                f"{len(wave0)} wave-0 profile job(s)), not including these",
                "artifacts under this workstream were written by job(s) " + ", ".join(extra),
                f"tests/fixtures/regression/join/step6/{name}/",
            )
        P("")

    # ---------------------------------------------------------------- escalations
    P("## Decisions taken, decisions pending")
    P("")
    if dash.get("decisions"):
        P("Decisions already taken and recorded, verbatim:")
        P("")
        L.extend(
            tbl(
                [
                    [
                        d.get("date", "—"),
                        d.get("by", "—"),
                        " ".join(str(v) for k, v in d.items() if k not in ("date", "by")),
                    ]
                    if isinstance(d, dict)
                    else ["—", "—", str(d)]
                    for d in dash["decisions"]
                ],
                ["date", "by", "decision"],
            )
        )
    P("")
    P("Extracted from the dashboard's own `status`, `notes` and review verdicts: every sentence that")
    P("records an escalation, a held commit, a pending decision or an open question. Verbatim, so the")
    P("words stay the coordinator's rather than becoming a paraphrase of them.")
    P("")
    esc_rows = []
    for w in ws_list:
        name = w["workstream"]
        st = str(w.get("status") or "")
        if any(k in st.lower() for k in ESCALATION_WORDS):
            esc_rows.append([f"`{name}`", "status", st])
        blobs = [("note", t) for t in (w.get("notes") or [])]
        if (w.get("review") or {}).get("verdict"):
            blobs.append(("review", w["review"]["verdict"]))
        for kind, blob in blobs:
            for s in sentences(blob):
                if any(k in s.lower() for k in ESCALATION_WORDS):
                    esc_rows.append([f"`{name}`", kind, s])
    L.extend(tbl(esc_rows, ["workstream", "field", "what the dashboard records"]))
    if dash.get("incident_log"):
        P("Incident log:")
        P("")
        for e in dash["incident_log"]:
            P(f"- {esc(e)}")
        P("")
    if dash.get("oscar_capacity_note"):
        P(f"Cluster capacity during Step 6, as recorded: {esc(dash['oscar_capacity_note'])}.")
        P("")

    # ---------------------------------------------------------------- not done
    # ---------------------------------------------------------------- deferred / demonstrated
    P("## Deferred possible work")
    P("")
    P("Read from `dashboard.yaml: deferred_work` — items that were identified, costed or designed and")
    P("then not funded inside Step 6. They are listed so that the blocker table above is read as a")
    P("state of the work rather than as a ceiling: a cell below its tier with an unfunded design")
    P("behind it is a decision, not a physical limit.")
    P("")
    deferred = []
    for entry in dash.get("deferred_work") or []:
        if isinstance(entry, dict):
            deferred.append(["(project)", entry.get("item"), entry.get("reason"), entry.get("status", "deferred")])
        else:
            deferred.append(["(project)", entry, "—", "deferred"])
    for w in ws_list:
        for entry in w.get("deferred_work") or []:
            if isinstance(entry, dict):
                deferred.append(
                    [f"`{w['workstream']}`", entry.get("item"), entry.get("reason"), entry.get("status", "deferred")]
                )
            else:
                deferred.append([f"`{w['workstream']}`", entry, "—", "deferred"])
    if deferred:
        L.extend(tbl(deferred, ["where", "item", "reason", "status"]))
    else:
        P("**The dashboard records no `deferred_work` entries.** That is an empty field, not a proof")
        P("that the possibilities are exhausted. The candidates below were extracted from the same")
        P("notes and review verdicts by keyword, and none of them has a `deferred_work` entry yet:")
        P("")
        L.extend(tbl(candidate_deferred(ws_list), ["workstream", "field", "what the note says"]))
        P("Until the field is populated, this section under-reports what remains possible — and says")
        P("so rather than reading as complete.")
        P("")

    P("## Demonstrated blockers")
    P("")
    P("Where an attempt was made and the ceiling was measured rather than assumed. Read from the")
    P("dashboard's `decisions` entries that use the phrase, and from the workstream fields that do.")
    P("")
    demo = []
    for dec in dash.get("decisions") or []:
        txt = (
            " ".join(str(v) for k, v in dec.items() if k not in ("date", "by")) if isinstance(dec, dict) else str(dec)
        )
        if "demonstrated blocker" in txt.lower():
            who = f"{dec.get('by', '?')} {dec.get('date', '')}".strip() if isinstance(dec, dict) else "decision"
            demo.append([who, "`dashboard.yaml: decisions`", txt])
    for w in ws_list:
        for field in ("status", "goal"):
            v = str(w.get(field) or "")
            if "demonstrated blocker" in v.lower() or "demonstrated:" in v.lower():
                demo.append([f"`{w['workstream']}`", f"`{field}`", v])
        for note in w.get("notes") or []:
            for sent in sentences(note):
                if "demonstrated blocker" in sent.lower():
                    demo.append([f"`{w['workstream']}`", "`notes`", sent])
    if demo:
        L.extend(tbl(demo, ["where", "field", "what was demonstrated"]))
    else:
        P("_No `decisions` entry and no workstream field uses the phrase. Nothing on this page is a")
        P("demonstrated ceiling._")
        P("")

    P("## What Step 6 did not do")
    P("")
    P("Derived from the dashboard's `status`, `authorization`, `performance` and `correctness` fields")
    P("and from the configuration table above — not from a list maintained beside them, which is the")
    P("kind of list that goes stale the first time a workstream lands. Each line names the field it")
    P("was read from.")
    P("")
    nd = []
    for w in ws_list:
        name, st = w["workstream"], str(w.get("status", ""))
        low = st.lower()
        perf, corr = w.get("performance") or {}, w.get("correctness") or {}
        if low.startswith("blocked"):
            nd.append([f"`{name}`", "status", f"not resolved — {st}"])
        elif low.startswith("partially merged") or (low.startswith("merged") and "through" in low):
            nd.append([f"`{name}`", "status", f"merged in part only — {st}"])
        elif not low.startswith("merged"):
            nd.append([f"`{name}`", "status", f"not merged — {st}"])
        if "held" in low or "held" in str(w.get("head_sha", "")).lower():
            nd.append(
                [
                    f"`{name}`",
                    "status / head_sha",
                    f"work delivered and held rather than merged; branch head `{w.get('head_sha')}`",
                ]
            )
        if str(w.get("authorization", "")).upper().startswith(("PROFILE ONLY", "DESIGN")):
            nd.append([f"`{name}`", "authorization", f"never authorized to implement — `{w.get('authorization')}`"])
        if perf.get("cpu_before") and not perf.get("cpu_after"):
            nd.append([f"`{name}`", "performance.cpu_after", "no after-measurement on CPU"])
        if perf.get("gpu_before") and not perf.get("gpu_after"):
            nd.append([f"`{name}`", "performance.gpu_after", "no after-measurement on GPU"])
        if str(corr.get("step4_replay", "")).strip().lower().startswith("n/a"):
            nd.append([f"`{name}`", "correctness.step4_replay", f"no Step-4 replay — {corr.get('step4_replay')}"])
    for dev, label in (("cpu", "CPU"), ("gpu", "GPU")):
        never = sorted(c for c in step4 if (c, dev) not in rep3 and not tier_obs.get((c, dev)))
        if never:
            nd.append(
                [
                    "(the corpus)",
                    "the table above",
                    f"{len(never)} frozen configuration(s) still carry no {label} measurement at "
                    f"all: " + ", ".join(f"`{c}`" for c in never),
                ]
            )
    L.extend(tbl(nd, ["workstream", "field it is read from", "what was not done"]))

    # ---------------------------------------------------------------- semantics
    P("## Project semantic status")
    P("")
    P("Step 5's parity-tier semantic gate PASSED, and Step 6 was not permitted to invalidate it. The")
    P("condition for carrying it forward is mechanical — no workstream whose code is on this branch")
    P("may have changed representative pixels or owe a semantic follow-up — so it is asserted here")
    P(f"rather than described. *Merged* below is decided by {merged_basis}.")
    P("")
    L.extend(
        tbl(
            [
                [
                    f"`{w['workstream']}`",
                    w.get("status"),
                    as_written(w.get("pixels_changed")),
                    as_written(w.get("semantic_followup")),
                ]
                for w in merged
            ],
            ["merged workstream", "status", "`pixels_changed`", "`semantic_followup`"],
        )
    )
    bad = [
        w["workstream"]
        for w in merged
        if first_word(w.get("pixels_changed")) != "no" or first_word(w.get("semantic_followup")) != "none"
    ]
    if bad:
        P(f"**The assertion FAILS.** {', '.join('`' + b + '`' for b in bad)} merged without")
        P("`pixels_changed: no` and `semantic_followup: none`, so the Step-5 gate cannot be carried")
        P("forward unexamined and the follow-up section 16 requires is outstanding.")
    else:
        P(f"**All {len(merged)} workstreams with code on this branch record `pixels_changed: no`**")
        P("**and `semantic_followup: none`.** No merged Step-6 change moved a representative pixel, so")
        P("nothing reaches the surfaces Step 5 measured and the Step-5 parity-tier gate stands exactly")
        P("as Step 5 left it. That is a statement about the merged set only: the held and unmerged commits")
        P("listed above have not been through this check, and the pixel comparisons each workstream")
        P("committed are tabulated per workstream so the claim can be audited against them.")
    P("")
    rest = [w for w in ws_list if w not in merged]
    if rest:
        P("The workstreams with nothing on this branch, for completeness. Their claims are not part of")
        P("the assertion above:")
        P("")
        L.extend(
            tbl(
                [
                    [
                        f"`{w['workstream']}`",
                        w.get("status"),
                        as_written(w.get("pixels_changed")),
                        as_written(w.get("semantic_followup")),
                    ]
                    for w in rest
                ],
                ["workstream", "status", "`pixels_changed`", "`semantic_followup`"],
            )
        )

    # ---------------------------------------------------------------- flags
    P("## Where the dashboard and the artifacts disagree")
    P("")
    if flags:
        P("Each row is a place where a committed artifact says something the dashboard does not. The")
        P("page prints the artifact value everywhere above; the rows are listed here so the dashboard")
        P("gets corrected, rather than the page acquiring a special case for each one.")
        P("")
    # Two artifacts can say the same thing about the same cell — a second sweep on a second card, for
    # instance. That is one disagreement with two sources, not two disagreements, and rendering it as
    # two would make the table look worse than the evidence does.
    grouped: dict[tuple[str, str, str], list[str]] = {}
    for where, says, art, path in flags:
        grouped.setdefault((where, says, art), []).append(path)
    L.extend(
        tbl(
            [[w, d_, a, ", ".join(dict.fromkeys(paths))] for (w, d_, a), paths in grouped.items()],
            ["where", "the dashboard says", "the artifact says", "artifact"],
        )
    )

    # ---------------------------------------------------------------- appendix
    P("## Appendix — artifacts not summarised")
    P("")
    P(f"{len(A.unknown)} JSON file(s) under `tests/fixtures/regression/join/step6/` are in a shape this generator")
    P("does not recognise. They are named rather than dropped: an artifact nothing reads is an")
    P("artifact nobody is checking, and this list is the to-do for the next run.")
    P("")
    L.extend(tbl([[f"`{p}`", why] for p, why in A.unknown], ["artifact", "why it is not summarised"]))
    if A.superseded or A.superseded_other:
        P(f"{len(A.superseded)} artifact(s) were **withdrawn by the workstream that took them** — moved into a")
        P("`superseded_*` directory, or stamped `superseded` / `unusable` — and nothing on this page")
        P("is computed from them. They are listed with the reason rather than deleted, because a")
        P("measurement that was taken and then not stood behind is part of the record:")
        P("")
        L.extend(tbl([[f"`{p}`", why] for p, why in A.superseded], ["artifact", "why it was withdrawn"]))
        if A.superseded_other:
            P(f"A further {A.superseded_other} non-JSON file(s) sit beside them in those directories.")
            P("")
    # Recognised is not the same as used. Anything whose path has not appeared anywhere on the page by
    # now is something the report read and then rested nothing on, and saying so is the difference
    # between an artifact that was considered and one that was skipped.
    so_far = "\n".join(L)
    quiet = [(path, kind) for path, kind in sorted(A.kind.items()) if f"`{path}`" not in so_far]
    if quiet:
        P(f"A further {len(quiet)} artifact(s) were recognised and read, but nothing on this page")
        P("rests on them. They are the inputs to numbers that appear here in reduced form, or")
        P("measurements a workstream took that no section above needed:")
        P("")
        L.extend(tbl([[f"`{p}`", kind] for p, kind in quiet], ["artifact", "what it is"]))
    P(
        f"Read and summarised: {len(A.profiles)} profile note(s), {len(A.tier_deltas)} tier artifact(s), "
        f"{len(A.sweeps)} Step-3 benchmark shard(s), {len(A.certs)} Gate-A certification(s), "
        f"{len(A.conformance)} conformance shard(s), {len(A.replay_legs)} replay leg(s) and "
        f"{len(A.pixel)} pixel comparison(s). A further {len(A.other_files)} non-JSON file(s) — "
        f"findings, the Markdown twin of each profile note, state banks, test logs — are evidence "
        f"behind the prose above and are not parsed here."
    )
    P("")
    return scrub("\n".join(L).rstrip() + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=ROOT / "tests" / "fixtures" / "regression" / "join" / "step6")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "research" / "step6_optimization.md")
    ap.add_argument("--repo", type=Path, default=ROOT, help="the tree whose SHAs and merge log are reported")
    ap.add_argument("--baseline-sha", default=None, help="override; the dashboard's baseline_sha is used otherwise")
    ap.add_argument(
        "--integration-sha", default=None, help="override for a copy with no .git (the Oscar rsync copies have none)"
    )
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(build(a.results, a.repo, a))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

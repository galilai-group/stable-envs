# Private OGBench Maze process candidate

Base 69ac94c20d859ffc6d55911654fd85f53169ae63 plus the source-hashed candidate.
The public factory remains local. Bounded spawn processes own independent source
worlds and isolated RNGs. Typed shared columns carry complete State and nested
task results; parent copies own outputs before the next request. Rendering stays
in the parent. Single actions and chunks share the existing slice-only API.

Restore prepares and validates all shards before any owner commits. Expected
validation errors discard preparation and leave owners usable. Unexpected worker
failure closes and reaps all owners. No Cube/Scene process implementation is
claimed: those families require actual staged replacement worlds and explicit
parent metadata instead of batch-sized duplicate simulators.

Local tests: nine passed, two skipped, four deselected. The skips are CUDA export
and the public GL constructor unavailable on sandboxed macOS. Tests include the
independent source oracle, full RNG/cache continuation, mixed active lanes,
retained output ownership, invalid late-shard restore, in-flight request exclusion
and worker death. Public GL/CPU/CUDA acceptance is job 6964833, pending at snapshot.
It also measures local/P4/P8 at B64/256 and T1/8 with six samples per timing cell.
No default change or GPU acceptance yet.

oscar/source_manifest.json pins the 171-file submitted bundle. Rebuild files from
this commit, applying source_overrides.json: it retains the previous shared
masking implementation, because the sparse-row optimization was developed after
submission. The new mask has separate native/Fetch gates and needs subsequent
integration coverage before a process default is accepted.

The same local source-oracle/adapter suite was rerun after sparse-padding
integration: nine passed, two skipped, four deselected; separate JUnit preserved.

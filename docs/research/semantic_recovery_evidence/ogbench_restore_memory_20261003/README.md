# Scene restore memory diagnostic

The B64 paired Scene default gate exceeded its 64 GiB allocation. This bounded
CPU-only diagnostic investigates repeated reconstruction without rendering in
two fresh Mac processes, with and without explicit collection after each restore.
Both use the current accepted adapter and pass exact full-State recapture after
all twelve restores. Each B2 State is 20,970 bytes.

| Process | Before construction | Constructed | Final restore | After final collection |
| --- | ---: | ---: | ---: | ---: |
| Ordinary collection | 323.7 MiB | 963.6 MiB | 1,341.0 MiB | 1,335.7 MiB |
| Collection after each restore | 324.0 MiB | 942.9 MiB | 1,254.5 MiB | 1,255.3 MiB |

The later restore samples plateau. Explicit collection reduces this working set
modestly; it does not eliminate the large simulator/model footprint. There is
no evidence of unbounded restore-only growth in this small run. This does not
prove the Linux B64 case, rendering-resource behavior, or larger-batch safety.
No collection call or memory workaround was added to production.

The paired acceptance gate keeps both a local B64 world set and a process-owned
B64 set, and transactional restore temporarily builds replacement worlds before
discarding the old set. This supplies a concrete working-set explanation to
investigate, rather than attributing the OOM to the small returned State. The
unchanged Scene gate is retried separately with 128 GiB; its real reports remain
required. Future scaling must report simulator memory as well as State bytes.

Raw reports retain RSS, GC counts, tracked objects, restore duration, dependency
versions and source hashes. `ogbench_restore_memory.py` uses `ps` only for its own
process RSS. The local sandbox denied that read, so these successful CPU runs
used authorized execution outside that restriction. Initial import/read failures
produced no successful measurements and are not part of the two results above.

# Round 1 — superseded

Jobs `6346291` (CPU, `node2344`, load average **43.79**) and `6346300` (GPU, `gpu2105`).

Two reasons these are not the authoritative shards:

1. **They answer a different question.** Their `before` leg is the Step-5 baseline with only the
   device-dispatch fix, so they measure RASTER2D-2 and SWMNAV-1 *and* the Step-6 bounded rasterizer
   together. The shards in the parent directory measure SWMNAV-1 against `1103326`. Counted side by
   side, one env would carry two tiers.
2. **Load average 43.79 on a 48-core node**, against 40.3 for the round-3 re-measurement. Both legs
   ran in one job so the ratio is still sound, but the absolute f/s are the noisier pair.

They are kept rather than deleted because the comparison against the Step-5 baseline — PushT
115 -> 590 f/s, SimplePointMaze 3 -> 112 f/s — is the one the findings quote as the workstream's
headline, and that number has to have a file behind it.

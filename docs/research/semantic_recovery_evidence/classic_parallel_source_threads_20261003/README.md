# Eight-thread CPU renderer versus serial/parallel source

Array 6971892 repeats all five classic comparisons at B64/256, T1/8, and
no/final/all frames. The parent Torch renderer receives eight CPU threads on
CPU and one on CUDA; each of eight persistent source workers remains single
threaded. Every strategy has the same eight-core allocation. Source pygame
has no GPU renderer: CUDA-output source timings include CPU rendering and
delivery. All source State/task/image and ownership gates remain required.

This corrects the CPU underutilization demonstrated by the separate thread
profile. It changes benchmark configuration, not production routing or code.
The one-thread campaign is retained unchanged. All five old CSVs reproduce
byte-for-byte with the updated summary's default one-thread mode.

The local frozen Pendulum B4/P2/two-thread CPU smoke passes six independent
source comparisons with zero image MSE. Full cluster outputs have not yet been
collected; no new cluster source speedup is claimed. Candidate reconstruction:
base 061e7a29 plus the two updated parallel-source benchmark/summary scripts,
verified by candidate_manifest.json. Source wrappers remain pinned at
176d54532b3c0a251a4f509b74f2fdacc143a63d.

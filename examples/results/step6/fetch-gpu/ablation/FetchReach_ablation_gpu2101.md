# Context-flag ablation — FetchReach on NVIDIA GeForce RTX 3090

host gpu2101.oscar.ccv.brown.edu  ·  slurm 6344691  ·  frozen states `/oscar/scratch/sboughan/se6/fetch-gpu/examples/results/step4/states/FetchReach.npz`

| cell | fps(1) | fps(256) | fps(4096) | us/frame(max B) | vs control | BATCH_SCALES | pixels identical |
|---|---|---|---|---|---|---|---|
| control | 602 | 5138 | 5214 | 191.8 | 1.000x | 8.7x | YES |
| skybox_off | 616 | 5291 | 5386 | 185.7 | 1.033x | 8.7x | YES |
| specular_off | 614 | 5185 | 5279 | 189.4 | 1.013x | 8.6x | NO |
| textures_off | 618 | 5202 | 5301 | 188.6 | 1.017x | 8.6x | YES |
| lightspec | 606 | 4861 | 4922 | 203.2 | 0.944x | 8.1x | YES |
| chunk_512 | 627 | 5094 | 5214 | 191.8 | 1.000x | 8.3x | YES |
| chunk_1024 | 638 | 5104 | 5216 | 191.7 | 1.000x | 8.2x | YES |
| skybox+textures_off | 595 | 5297 | 5407 | 185.0 | 1.037x | 9.1x | YES |
| skybox+textures_off+chunk512 | 628 | 5307 | 5422 | 184.4 | 1.040x | 8.6x | YES |

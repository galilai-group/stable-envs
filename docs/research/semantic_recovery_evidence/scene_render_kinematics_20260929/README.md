# Scene render-only geometry update gate

Builds on the independently validated wrapper/duplicate-forward cleanup. The baseline is that cleanup (V1), candidate is the guarded kinematics update (V2). Changes only Scene renderer-owned data and visual colors, no simulation, State schema, shaders, or quality settings. Unsupported visualization/model features retain full-forward fallback.

RTX3090 job6806327 passed paired B64/128/256 and both output destinations: full-State nonmutation, same saved16States and same host, source MSE <=0.001 for every sample, baseline/candidate images within1uint8. B256 CPU output:1355.787FPS versus source1370.200FPS (0.98948x, approximately a tie); CUDA output:1375.656 versus1355.712FPS (1.01471x). This is not a large source-relative speedup.

CPU job 6806329 completed and passed; the tested V2 is adopted for both devices. Supplemental A5000/A5500 results stay separate. Raw timing samples and SD, full-batch images, scripts, and hashes are retained.

Timing boundary: source States and MuJoCo derived geometry are installed before timing; source public render is executed for every image. Our public render accepts RenderState and computes geometry inside its timer. Neither side loads disk States or performs rollouts inside timing. These are public renderer throughput measurements against a prepared source; they are not a matched benchmark of full-State installation plus rendering.

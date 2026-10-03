# Same benchmark, available installer

V2 job 6968409 failed before setup because se-step6 has no pip. Its terminal log
is preserved under ../failed-v2. V3 uses the existing /users/sboughan/.local/bin/uv
with --python se-step6/bin/python, --no-deps and a fresh --target scratch directory.
The same mujoco-warp 3.12.0 and warp-lang 1.17.0 pins are used. No accepted venv
is modified. The benchmark and all 170 frozen source files are unchanged from V2.
The installation profile records package versions, requirements and SHA256 for
installed files. CPU and CUDA both rerun; no old partial measurements are spliced.

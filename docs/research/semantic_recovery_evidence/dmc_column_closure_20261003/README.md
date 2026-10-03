# DMC exporter closure: eleven complete pairs, one unresolved failure

Array 6961949 is terminal. Eleven registrations independently pass CPU software-GL
and CUDA/EGL State/task/RNG/lifecycle/ownership and exact-frame gates. The strict
summary contains 264 paired cells with six samples each. Quadruped passes CPU but
fails the CUDA frame equality assertion; its original FAIL report and log are
preserved and it is excluded from the paired CSV. No all-twelve acceptance.

Candidate source is reconstructed from c291296c642090dc77f86902b2f7d55afcb38d1d;
baseline adapters from ce3ec1ffd01698409fe14a7c6dc88a78a207e9a0. Report candidate,
reference and benchmark hashes were checked against those exact files. All GPUs
were RTX 3090. CPU physics is used for both output devices. These are exporter
comparisons against the previous library version, not upstream speedups or FPS.

Diagnostic 6964907 runs the unchanged frozen Quadruped benchmark on its original
host gpu2109, adding State byte comparisons, frame MSE and repeated rendering of
the same owned States. The driver and launcher are under diagnostic/. A failure
to reproduce does not resolve the original failure or qualify as acceptance.

Reproduce the selected summary with summarize_dmc_state_export.py, passing
--source for the frozen candidate, --reference-adapters for the frozen baseline,
and --domains acrobot pendulum cartpole cheetah walker hopper ball_in_cup finger
reacher manipulator humanoid. Omitting --domains includes Quadruped and must fail.
Source reconstruction and digest assertions are in source-provenance.json.

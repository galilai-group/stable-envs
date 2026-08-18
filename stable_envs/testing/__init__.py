"""Checks you can run on YOUR env, plus the harness that runs ours. Not the same thing as ``tests/``.

Two directories in this repository have "test" in the name and they are not duplicates:

  ``stable_envs/testing/``  (here) is LIBRARY code, importable and public, in the same sense as
                            ``numpy.testing`` or ``torch.testing``. Nothing in it is specific to this
                            repo's own environments.
  ``tests/``                is this repo's own pytest suite — the assertions, the thresholds and the
                            per-env measured floors. It imports from here; nothing here imports from
                            it, and nothing here is needed to USE stable-envs.

What is in here splits again, by audience, and the split is worth knowing before you import anything:

  **The checks.** ``conformance`` is the surface an env author calls. Every function in it takes an
  env and returns a dict of measurements — ``check_state_roundtrip``, ``check_batch_invariance``,
  ``check_state_dims_matter``, ``pixel_agreement``, ``check_resolution_consistency`` and the rest —
  and NONE of them asserts anything. That is deliberate: the numbers are the product, the thresholds
  belong to whoever is judging, and ``tests/gpu/test_conformance.py`` is one caller of this API rather
  than its definition. ``docs/adding-an-env.md`` and nine scripts under ``examples/`` use it exactly
  that way, on envs that are not in this registry. If you are writing an env, this is the module you
  want and you can stop reading here.

  **The harness.** ``supervisor``, ``safety_plugin``, ``resource_guard``, ``node_health`` and
  ``render_worker`` exist because running a large GPU suite on this cluster is itself a problem: a
  MuJoCo or EGL fault takes the interpreter down and a bare pytest then reports nothing at all, an
  env that releases an EGL context corrupts every other live renderer in the process, and a node the
  scheduler has already condemned produces parity numbers that mean nothing. They are entered from
  outside — ``python -m stable_envs.testing.supervisor -- tests/ -v`` is what both sbatch scripts
  run, and ``conftest.py`` loads ``safety_plugin`` — rather than imported by anything you would write.
  ``render_worker`` is the odd one: it is a ``__main__`` that renders the registry in a FRESH
  interpreter, and ``tests/gpu/test_reproducibility.py`` imports its seed constants on purpose, so
  that both sides of a cross-process comparison provably draw the same inputs.

See ``docs/api/testing.md`` for the reference and ``docs/testing.md`` for how to run the suite.
"""

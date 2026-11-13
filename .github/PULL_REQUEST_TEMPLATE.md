## What and why

<!-- What does this change do, and why is it needed? -->

Closes #

## Tests run

<!-- Tick what you ran. Name the family profiles, for example dmc, pusht. -->

- [ ] `uvx nox -s "test(core)"`
- [ ] `uvx nox -s "test(<profile>)"` for each family this touches:
- [ ] `pytest tests/gpu` on a CUDA machine, if GPU code changed
- [ ] `uvx nox -s docs`, if docs changed

## Checklist

- [ ] Docs updated, or not needed.
- [ ] A line added to `CHANGELOG.md` under `Unreleased`, or not needed.
- [ ] No test tolerance, threshold or frozen fixture was loosened without discussion with a maintainer.

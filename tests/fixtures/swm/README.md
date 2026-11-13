# stable-worldmodel correspondence

`frozen_cells.json` maps each of the 38 stable-worldmodel ids to the `stable_envs` classes that implement it.
`tests/helpers/reference/crosswalk.py` reads it; `tests/unit/test_registry.py` and
`tests/integration/test_worldmodel.py` check the registry and bridge against it.

# packaging-ci (moved)

Public CI no longer invokes this directory. Canonical copies:

| here | product path |
|---|---|
| `test_packaging.py` | `tests/packaging/test_distribution.py` |
| `audit_dist.py` | `scripts/ci/audit_dist.py` |
| `smoke_installed.py` | `scripts/ci/smoke_installed.py` |
| `audit_optional.py` | `scripts/ci/audit_optional.py` |

The `.py` files here are thin wrappers so historical Oscar notes still resolve.
Ordinary GitHub Actions must call only the product paths.

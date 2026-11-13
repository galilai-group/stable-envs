import importlib.metadata

import pytest
import torch

from tests.helpers import ogbench_golden as golden

pytestmark = [pytest.mark.gpu, pytest.mark.regression]


@pytest.mark.parametrize("family,backend", golden.CASES)
def test_public_render_matches_golden(family, backend):
    if not torch.cuda.is_available():
        pytest.skip("Cube/Scene golden frames require CUDA")
    for package, expected in golden.VERSIONS.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pytest.skip(f"Cube/Scene golden frames require {package}=={expected}")
        if actual != expected:
            pytest.skip(f"Cube/Scene golden frames require {package}=={expected}; found {actual}")
    manifest = golden.metadata()
    if torch.cuda.get_device_name() != manifest["gpu"]:
        pytest.skip(f"Cube/Scene golden frames require {manifest['gpu']}")

    from stable_envs.envs.ogbench import OGBCube, OGBScene

    state, expected = golden.load(golden.FIXTURES / f"{family}-{backend}.npz")
    cls = OGBCube if family == "Cube" else OGBScene
    with cls(device="cuda", size=64, seed=17) as env:
        actual = env.render(state, backend=backend).cpu().numpy()
    golden.assert_frames(actual, expected)
    # A one-channel rendering regression must fail the same comparison.
    actual[0, 0, 0, 0] ^= 1
    with pytest.raises(AssertionError):
        golden.assert_frames(actual, expected)

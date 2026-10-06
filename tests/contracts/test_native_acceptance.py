import pytest

from tests.helpers.reference.native_acceptance import (
    _continuous_dtype_flag_necessity,
    _physical_variation,
    run_case,
)
from tests.helpers.reference.profiles import PROFILES

pytestmark = [pytest.mark.integration, pytest.mark.reference]
DOMAINS = [case.domain for case in PROFILES["native"]]


@pytest.fixture(autouse=True)
def native_source(request, source_identity):
    if request.config.getoption("--contract-profile") != "native":
        pytest.skip("Specialized native witnesses belong to the native profile")


@pytest.mark.parametrize(
    "domain,batch", [(d, b) for d in DOMAINS for b in (1, 3)], ids=[f"{d}-B{b}" for d in DOMAINS for b in (1, 3)]
)
def test_native_boundaries(domain, batch, request):
    run_case(domain, batch, request.config.getoption("--contract-device"))


@pytest.mark.parametrize("domain", DOMAINS)
def test_native_physics_variation(domain, request):
    _physical_variation(domain, request.config.getoption("--contract-device"))


def test_native_dtype_control(request):
    _continuous_dtype_flag_necessity(request.config.getoption("--contract-device"))

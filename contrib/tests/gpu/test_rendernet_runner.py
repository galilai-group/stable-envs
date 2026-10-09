import pytest
import torch

from contrib.render import rendernet

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.mark.parametrize("network", [rendernet.ESPCN, rendernet.UNetTiny])
@pytest.mark.parametrize("mode", ["eager", "default"])
def test_runner_preserves_frames_across_replays_and_padding(network, mode):
    torch.manual_seed(17)
    module = network(in_size=8, out_size=16, width=4, depth=2)
    with torch.no_grad():
        module.head.weight.normal_(std=0.01)
    module = module.eval().cuda().half().to(memory_format=torch.channels_last)
    module.requires_grad_(False)
    runner = rendernet._Runner(module, 4, 8, 16, "cuda", mode)
    try:
        low = torch.randint(256, (4, 8, 8, 3), device="cuda", dtype=torch.uint8)
        expected = runner(low).clone()
        runner(torch.zeros_like(low))
        assert torch.equal(runner(low), expected)
        assert torch.equal(runner(low[:1]), expected[:1])
        permutation = torch.tensor([2, 0, 3, 1], device="cuda")
        assert torch.equal(runner(low[permutation]), expected[permutation])
    finally:
        runner.close()

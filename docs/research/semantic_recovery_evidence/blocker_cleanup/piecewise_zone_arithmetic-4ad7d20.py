"""Isolate the observed Piecewise x=112 CUDA zone lookup; no production edits."""
import json
import torch

result = {'torch': torch.__version__, 'cuda': torch.version.cuda, 'devices': {}}
for device in ('cpu', 'cuda'):
    x = torch.tensor([111.99999, 112., 112.00001], dtype=torch.float32, device=device)
    numerator = x.double() - 14.
    scalar_quotient = numerator / 196.
    tensor_quotient = numerator / torch.full_like(numerator, 196.)
    python_quotient = [(float(v) - 14.) / 196. for v in x.cpu()]
    original_index = (scalar_quotient * 2).trunc().long()
    tensor_index = (tensor_quotient * 2).trunc().long()
    result['devices'][device] = {
        'positions': x.cpu().tolist(),
        'numerator': numerator.cpu().tolist(),
        'scalar_division_hex': [float(v).hex() for v in scalar_quotient.cpu()],
        'tensor_division_hex': [float(v).hex() for v in tensor_quotient.cpu()],
        'python_division_hex': [v.hex() for v in python_quotient],
        'original_index': original_index.cpu().tolist(),
        'tensor_denominator_index': tensor_index.cpu().tolist(),
        'source_index': [int(v*2) for v in python_quotient],
    }
print(json.dumps(result, indent=2))

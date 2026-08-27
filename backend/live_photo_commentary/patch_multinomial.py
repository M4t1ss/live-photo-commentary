"""
Monkey-patches torch.multinomial to route MPS-device sampling through CPU.

torch.multinomial(probs, 1) on the MPS backend can silently sample an index
with exactly zero probability. Root cause (see
https://github.com/pytorch/pytorch/issues/192577): MPS's exponential_(1),
used internally by multinomial's single-sample Gumbel fast path, sometimes
produces an exact zero that surfaces as -0.0; dividing by that negative
zero yields NaN, and MPS's argmax treats NaN as the maximum, picking the
corrupted (zero-probability) index. CPU sampling doesn't hit this path and
isn't affected. This patch forces just that one op onto CPU, leaving
everything else (the forward pass, etc.) running on MPS as normal.

Import this before calling model.generate():

    from . import patch_multinomial  # noqa: F401
"""

# This runs before local_describer.py, which normally guards the first
# `import torch` against a Windows NTFS staleness race after a CUDA
# reinstall — see _wait_for_torch.py.
from ._wait_for_torch import wait_for_torch
wait_for_torch()

import torch


_real_multinomial = torch.multinomial

def _mps_safe_multinomial(input, num_samples, replacement=False, *, generator=None, out=None):
    if input.device.type == "mps":
        return _real_multinomial(
            input.cpu(), num_samples, replacement=replacement, generator=generator
        ).to(input.device)
    return _real_multinomial(
        input, num_samples, replacement=replacement, generator=generator, out=out
    )

torch.multinomial = _mps_safe_multinomial

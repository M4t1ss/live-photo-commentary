"""
Monkey-patches torch.multinomial to route non-CPU-device sampling through CPU.

torch.multinomial on non-CPU backends has been observed to occasionally
sample an index with exactly zero probability after top-k filtering --
the same distribution sampled on CPU did not reproduce this. This patch
forces just that one op off the GPU, leaving everything else (the forward
pass, etc.) running on GPU as normal.

Import this before calling model.generate():

    from . import patch_multinomial  # noqa: F401
"""

import torch


_real_multinomial = torch.multinomial

def _cpu_multinomial(input, num_samples, replacement=False, *, generator=None, out=None):
    if input.device.type != "cpu":
        return _real_multinomial(
            input.cpu(), num_samples, replacement=replacement, generator=generator
        ).to(input.device)
    return _real_multinomial(
        input, num_samples, replacement=replacement, generator=generator, out=out
    )

torch.multinomial = _cpu_multinomial

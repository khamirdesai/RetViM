"""Numerical checks for the full selective recurrence, gating and causality."""

import pytest
import torch
from torch.nn import functional as F

from retvim.model import NativeMamba, selective_scan


def reference_scan(u, delta, a, b, c, d, z):
    """Independent einsum formulation of the published Mamba-1 equations."""
    dtype = u.dtype
    u, delta, b, c = u.float(), delta.float(), b.float(), c.float()
    transition = torch.exp(torch.einsum("bld,dn->bldn", delta, a.float()))
    drive = torch.einsum("bld,bln,bld->bldn", delta, b, u)
    state = a.new_zeros((u.shape[0], a.shape[0], a.shape[1]))
    result = []
    for index in range(u.shape[1]):
        state = transition[:, index] * state + drive[:, index]
        result.append(torch.einsum("bdn,bn->bd", state, c[:, index]))
    return ((torch.stack(result, dim=1) + u * d.float()) * F.silu(z.float())).to(dtype)


@pytest.mark.parametrize("length", [1, 7, 33, 65])
def test_selective_scan_matches_reference_across_chunks(length):
    generator = torch.Generator().manual_seed(87)
    u = torch.randn(2, length, 6, generator=generator)
    delta = F.softplus(torch.randn(2, length, 6, generator=generator))
    a = -torch.rand(6, 4, generator=generator)
    b = torch.randn(2, length, 4, generator=generator)
    c = torch.randn(2, length, 4, generator=generator)
    d = torch.randn(6, generator=generator)
    z = torch.randn(2, length, 6, generator=generator)
    actual = selective_scan(u, delta, a, b, c, d, z)
    expected = reference_scan(u, delta, a, b, c, d, z)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    # Catch the legacy website's omitted recurrence, which kept only this term.
    assert not torch.allclose(actual, u * d * F.silu(z))


def test_native_mamba_matches_full_reference_forward():
    torch.manual_seed(23)
    layer = NativeMamba(dim=8, d_state=3, d_conv=4).eval()
    tokens = torch.randn(2, 35, 8)
    x, gate = F.linear(tokens, layer.in_proj.weight).chunk(2, dim=-1)
    x = F.conv1d(F.pad(x.transpose(1, 2), (3, 0)), layer.conv1d.weight,
                 layer.conv1d.bias, groups=16).transpose(1, 2)
    x = F.silu(x)
    delta, b, c = F.linear(x, layer.x_proj.weight).split([1, 3, 3], dim=-1)
    delta = F.softplus(F.linear(delta, layer.dt_proj.weight) + layer.dt_proj.bias)
    expected = F.linear(reference_scan(x, delta, -layer.A_log.exp(), b, c,
                                       layer.D, gate), layer.out_proj.weight)
    torch.testing.assert_close(layer(tokens), expected, rtol=2e-6, atol=2e-6)


def test_native_mamba_is_causal():
    torch.manual_seed(42)
    layer = NativeMamba(dim=8, d_state=3).eval()
    tokens = torch.randn(1, 12, 8)
    modified = tokens.clone()
    modified[:, 6:] += 50
    torch.testing.assert_close(layer(tokens)[:, :6], layer(modified)[:, :6])

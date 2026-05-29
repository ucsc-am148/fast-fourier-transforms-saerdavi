"""STUDENT FILE: implement the canonical twiddle helpers.

Three patterns (so seven inconsistently-named lecture helpers collapse to ~3):
  1. radix-2 length-N/2 twiddles   make_radix2_twiddles
  2. per-stage radix-16 twiddles   make_radix16_twiddles
  3. Bailey cross-term twiddles    make_bailey_cross_twiddles

Plus two scaffolding tables (full DFT, padded-R DFT) and the bit-reversal
permutation. Use the forward-FFT sign convention exp(-2*pi*i * ...) and
return (re, im) tuples of separate real-valued tensors everywhere.

When you implement each function, the signature should match the docstring
exactly -- the harness expects (re, im) tuples with specific shapes/dtypes,
and sanity_check.py will FAIL if you return something else.
"""

import math

import torch


# =============================================================================
# Pattern 1: radix-2 length-N/2 twiddles  (F2, F3)
# =============================================================================

def make_radix2_twiddles(
    N: int,
    dtype: torch.dtype = torch.float32,
    device: str = 'cuda',
) -> tuple[torch.Tensor, torch.Tensor]:
    """w_N^k for k in [0, N/2). Returns (tw_re, tw_im), each shape (N//2,).

    Used by the radix-2 butterfly: stage s reads twiddle at index
    (k & (2**s - 1)) * (N >> (s+1)), so the table only needs the lower half
    of one full period."""
    
    k = torch.arange(N // 2, device=device, dtype=torch.float32)

    arg = -2.0 * math.pi * k / N

    tw_re = torch.cos(arg).to(dtype)
    tw_im = torch.sin(arg).to(dtype)

    return (tw_re, tw_im)


# =============================================================================
# Pattern 2: per-stage radix-16 twiddles  (F4; reused by F5/F6/F7 via F4)
# =============================================================================
# The index bookkeeping for this helper is given -- the per-stage permute
# schedule means the column-axis labels at stage s are a mix of already-
# transformed output digits and not-yet-transformed input digits, in an
# order set by the cumulative permutation history. 

def _column_axis_labeling(L: int) -> list[tuple]:
    """Track axis labels through the per-stage permute schedule.

    Convention: input n decomposes as n = sum_i d_i * 16^(L-1-i) with d_0 the
    high digit; output k similarly with e_i. Initial tile has axis i labeled
    ('d', i). At each stage s the kernel applies perm = (s,) + (others in
    original order), bringing axis s to position 0; the four-tl.dot then
    transforms position 0 from ('d', s) to ('e', L-1-s).

    Returns a list of length L; entry s is the tuple of L-1 labels at axis
    positions 1..L-1 of the (16,)*L tile *after* the stage-s permute.
    """
    A = [('d', i) for i in range(L)]
    out = []
    for s in range(L):
        P = [A[s]] + [A[i] for i in range(L) if i != s]
        out.append(tuple(P[1:]))
        A = [('e', L - 1 - s)] + P[1:]
    return out


def make_radix16_twiddles(
    N: int,
    device: str = 'cuda',
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-stage radix-16 Cooley-Tukey twiddles, stacked. Returns
    (tw_re, tw_im), each shape (L, 16, N//16) fp16. L = log_16(N).

    Stage-0 slice is ones (kernel skips the multiply on s == 0). Stage s > 0
    is built from the labeling above via:
        tw[m, c] = exp(-2*pi*i * m * t / 16^(s+1))
        t = sum_{j=0}^{s-1} e_{L-1-j}_value(c) * 16^j
    where e_{L-1-j}_value(c) reads the base-16 digit of c at the position
    given by _column_axis_labeling(L)[s].
    """
    L = int(round(math.log(N, 16)))

    C = N // 16

    tw_re = torch.ones((L, 16, C), device=device, dtype=torch.float32)
    tw_im = torch.zeros((L, 16, C), device=device, dtype=torch.float32)

    m = torch.arange(16, device=device, dtype=torch.float32)[:, None]
    c = torch.arange(C, device=device, dtype=torch.int64)

    labels_by_stage = _column_axis_labeling(L)

    for s in range(1, L):
        labels = labels_by_stage[s] #Read: _column_axis_labeling(L)[s]

        # t is reconstructed from the already-transformed e-digits
        t = torch.zeros((C,), device=device, dtype=torch.int64)

        for col_pos, label in enumerate(labels):
            kind, idx = label

            if kind == "e":
                # digit value at this column-axis position
                #
                # c indexes an (L-1)-dimensional base-16 tile.
                # col_pos = 0 is the most significant remaining column digit.
                shift = L - 2 - col_pos
                digit_val = (c // (16 ** shift)) % 16

                # e_{L-1} contributes 16^0,
                # e_{L-2} contributes 16^1, etc.
                power = L - 1 - idx
                t += digit_val * (16 ** power)

        arg = -2.0 * math.pi * m * t[None, :].to(torch.float32) / (16 ** (s + 1))

        tw_re[s, :, :] = torch.cos(arg) # Euler :)
        tw_im[s, :, :] = torch.sin(arg)

    return (tw_re.to(torch.float16), tw_im.to(torch.float16))

# =============================================================================
# Pattern 3: Bailey cross-term twiddles  (F3, F5, F6, F7)
# =============================================================================

def make_bailey_cross_twiddles(
    m0: int,
    M: int,
    N: int,
    dtype: torch.dtype = torch.float16,
    device: str = 'cuda',
) -> tuple[torch.Tensor, torch.Tensor]:
    """w_N^{n1 * kM} for n1 in [0, m0), kM in [0, M). Returns (re, im), each
    shape (m0, M).

    F3 calls this with dtype=torch.float32 (the radix-2 tier is fp32);
    F5/F6/F7 call it with dtype=torch.float16 (the tcFFT tier is fp16). The
    Bailey identity holds for any N >= m0 * M; in practice N == m0 * M.
    """
    
    n1 = torch.arange(m0, device=device, dtype=torch.float32)
    kM = torch.arange(M, device=device, dtype=torch.float32)

    arg = -2.0 * math.pi * n1[:, None] * kM[None, :] / N

    re = torch.cos(arg).to(dtype)
    im = torch.sin(arg).to(dtype)

    return (re, im)

# =============================================================================
# Scaffolding tables
# =============================================================================

def make_dft_matrix(
    N: int,
    dtype: torch.dtype = torch.float16,
    device: str = 'cuda',
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full (N, N) DFT matrix. Returns (W_re, W_im).

    W[j, k] = exp(-2*pi*i * j * k / N). Used by F1 (DFT-as-complex-matmul).
    """
    n = torch.arange(N, device=device, dtype=torch.float32)
    k = torch.arange(N, device=device, dtype=torch.float32)

    arg = -2.0 * math.pi * n[:, None] * k[None, :] / N

    W_re = torch.cos(arg).to(dtype)
    W_im = torch.sin(arg).to(dtype)

    return W_re, W_im


def make_dft_R_padded(
    R: int,
    device: str = 'cuda',
) -> tuple[torch.Tensor, torch.Tensor]:
    """Length-R DFT padded to (16, 16) fp16. Returns (M_re, M_im).

    Pad the length-R row to 16 with zeros, hit it with a (16, 16) matrix whose
    first R columns are F_R (rows wrap mod R), take the first R output rows.
    This makes the >=16x16 tl.dot requirement hold for all R in {2, 4, 8, 16}.
    """
    n = torch.arange(16, device=device, dtype=torch.float32)
    k = torch.arange(16, device=device, dtype=torch.float32)

    n_mod = torch.remainder(n, R)

    phase = -2.0 * math.pi * n_mod[:, None] * k[None, :] / R

    M_re = torch.cos(phase)
    M_im = torch.sin(phase)

    # Columns k >= R should not contribute.
    col_mask = k < R
    M_re = torch.where(col_mask[None, :], M_re, torch.zeros_like(M_re))
    M_im = torch.where(col_mask[None, :], M_im, torch.zeros_like(M_im))

    return (M_re.to(torch.float16), M_im.to(torch.float16))


def bit_reversal_perm(N: int, device: str = 'cuda') -> torch.Tensor:
    """Length-N bit-reversal permutation as a (N,) int32 tensor.

    rev[i] is the integer whose n_bits=log2(N) binary representation is i's
    bits in reversed order.
    """
    LOG2_N = int(math.log2(N))

    vals = torch.arange(N, device=device, dtype=torch.int64)
    rev = torch.zeros_like(vals)

    for i in range(LOG2_N):
        rev = (rev << 1) | ((vals >> i) & 1)

    return rev.to(torch.int32)

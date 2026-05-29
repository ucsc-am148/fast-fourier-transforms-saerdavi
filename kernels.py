"""STUDENT FILE: implement the Triton kernels and pipeline drivers.

You implement:
  - Six @triton.jit kernels: f1_kernel, f2_kernel, transpose_kernel,
    f4_kernel_L2, dft_kernel, bailey_scale_kernel.
  - The f1_launch and f2_launch grid-choice wrappers around them.
  - The pipeline drivers: f3_launch, f5_launch, _f6_rec, _f7_rec.
  - f6_factor: the chunk-recipe for F6/F7.

You do NOT implement (left given below):
  - The thin launch wrappers _transpose, _fft_chunk, _scale, _lookup_tw.
    These are mechanical "pick the grid and launch one kernel" helpers.
  - The tuning constants F4_L2_BLOCK_B, DFT_BLOCK_B, SCALE_BLOCK,
    TRANSPOSE_BLOCK.

The signatures below are the ones the harness calls -- your job is to fill
the bodies. When your code passes sanity_check.py, you're done.
"""

import math

import torch
import triton
import triton.language as tl


# Tunings -- GIVEN.
F4_L2_BLOCK_B = 2
DFT_BLOCK_B = 16
SCALE_BLOCK = 32
TRANSPOSE_BLOCK = 32


# =============================================================================
# Device-function helper: complex matmul
# =============================================================================
# Implement this once -- f1_kernel, f4_kernel_L2, and dft_kernel all call it.


@triton.jit
def _cdot(a_re, a_im, b_re, b_im):
    """Complex matmul Y = A @ B as four real tl.dot calls.

    Returns (y_re, y_im) in fp32 (out_dtype=tl.float32). Caller is responsible
    for any fp16 down-cast on store. Works at any matmul shape tl.dot accepts.

    Used by f1_kernel, f4_kernel_L2, and dft_kernel. Don't reimplement the
    four-tl.dot expansion at each call site -- implement once here, call
    everywhere.

    TODO: implement.
    """

    # implementation of ( Re(a) + i*Im(a) ) * ( Re(b) + i*Im(b) ) = Re(a)Re(b) - Im(a)Im(b) + i*( Re(a)Im(b) + Im(a)Re(b) )

    y_re = tl.dot(a_re, b_re, out_dtype = tl.float32) - tl.dot(a_im, b_im, out_dtype = tl.float32)

    y_im = tl.dot(a_re, b_im, out_dtype = tl.float32) + tl.dot(a_im, b_re, out_dtype = tl.float32)

    return (y_re, y_im)


# =============================================================================
# Chunk factorization for F6 / F7
# =============================================================================

def f6_factor(N: int) -> list[int]:
    """Factor N = 2^k into FFT chunks.

    Recipe: prefer 256-length chunks (radix-256, handled by f4_kernel_L2), then
    16-length (handled by dft_kernel via the padded radix-16 path), then a
    small leftover in {2, 4, 8} for the remaining bits. chunks[0] is the
    innermost (fastest) input axis. Examples:
        256 -> [256]                4096 -> [256, 16]
        65536 -> [256, 256]         1048576 -> [256, 256, 16]
        64 -> [16, 4]               2 -> [2]
    """
    k = int(math.log2(N))
    chunks = []

    # Prefer 256 = 2^8 chunks
    while k >= 8:
        chunks.append(256)
        k -= 8

    # Then prefer one 16 = 2^4 chunk if possible
    if k >= 4:
        chunks.append(16)
        k -= 4

    # Then one small leftover, if any
    if k > 0:
        chunks.append(1 << k)

    return chunks


f7_factor = f6_factor   # F7 reuses F6's chunk recipe


# =============================================================================
# F1: DFT as one dense complex matmul (four tl.dot)
# =============================================================================

@triton.jit
def f1_kernel(
    x_re_ptr, x_im_ptr,    # (B, N) fp16
    W_re_ptr, W_im_ptr,    # (N, N) fp16; W[n, k]
    y_re_ptr, y_im_ptr,    # (B, N) fp32
    B,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Y = X @ W^T as four (BLOCK_M, BLOCK_K) x (BLOCK_K, BLOCK_N) tl.dot calls.

    Y[b, n] = sum_k X[b, k] * W[n, k]. Load W in transposed access
    (W_T[k, n] = W[n, k]) so tl.dot reads it the way it wants.

    Use `_cdot(x_re, x_im, W_T_re, W_T_im)` for the per-block complex matmul;
    accumulate its fp32 output into `acc_re` / `acc_im`.

    Dtype contract (same as F4): loads are fp16, `tl.dot` runs with
    `out_dtype=tl.float32` (handled by `_cdot`), accumulator is fp32, store
    is fp32. Allocations in `f1_alloc` already match this -- x_re/x_im are
    fp16, y_re/y_im are fp32.

    TODO: implement.
    """
    
    # 1. define PIDs and offsets

    mpid = tl.program_id(0)
    npid = tl.program_id(1)

    m_offsets = mpid * BLOCK_M + tl.arange(0,BLOCK_M)
    n_offsets = npid * BLOCK_N + tl.arange(0,BLOCK_N)

    # 2. load in the real and imaginary submatrices for X (inputs) and W (DFT)

    acc_re = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    acc_im = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k in range(0, N, BLOCK_K):

        k_offsets = k + tl.arange(0, BLOCK_K)

        # load X --------


            # Re(x)

        x_re = tl.load(
            x_re_ptr + m_offsets[:, None] * N + k_offsets[None, :],
            mask=(m_offsets[:, None] < B)
               & (k_offsets[None, :] < N),
            other=0.0,
        )
        
            # Im(x)

        x_im = tl.load(
            x_im_ptr + m_offsets[:, None] * N + k_offsets[None, :],
            mask=(m_offsets[:, None] < B)
               & (k_offsets[None, :] < N),
            other=0.0,
        )

        # load W --------
        
            # Re(w)
        W_T_re = tl.load(
            W_re_ptr + n_offsets[None, :] * N + k_offsets[:, None],
            mask=(n_offsets[None, :] < N)
               & (k_offsets[:, None] < N),
            other=0.0,
        )
        
            # Im(w)
        W_T_im = tl.load(
            W_im_ptr + n_offsets[None, :] * N + k_offsets[:, None],
            mask= (n_offsets[None, :] < N) 
                & (k_offsets[:, None] < N), 
                other=0.0)


        #2.5 call _cdot and accumulate

        tile_re, tile_im = _cdot(x_re, x_im, W_T_re, W_T_im)

        acc_re += tile_re
        acc_im += tile_im

    # 3. store Y

        # real parts Re(y)

    tl.store(
        y_re_ptr + m_offsets[:, None] * N + n_offsets[None, :],
        acc_re,
        mask=(m_offsets[:, None] < B)
           & (n_offsets[None, :] < N),
    )

        # imaginary parts Im(y)

    tl.store(
        y_im_ptr + m_offsets[:, None] * N + n_offsets[None, :],
        acc_im,
        mask=(m_offsets[:, None] < B)
           & (n_offsets[None, :] < N),
    )



def f1_launch(x_re, x_im, W_re, W_im, y_re, y_im):
    """Grid: (cdiv(B, BLOCK_M), cdiv(N, BLOCK_N)). One program tiles a
    (BLOCK_M, BLOCK_N) output square. tl.dot needs all three dims >=16, so B
    should be >= 16.

    TODO: implement.
    """

    B = x_re.shape[0]
    N = x_re.shape[1]

    BLOCK_M = 16
    BLOCK_K = 32
    BLOCK_N = 16

    grid = (
        triton.cdiv(B, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    f1_kernel[grid](
        x_re, x_im,
        W_re, W_im,
        y_re, y_im,
        B,
        N,
        BLOCK_M,
        BLOCK_K,
        BLOCK_N,
    )


# =============================================================================
# F2: radix-2 Cooley-Tukey, single program per signal
# =============================================================================
# F3 reuses this kernel! For F2, only BAILEY_EPILOGUE=False, STRIDED_STORE=False need to be implemented.
#
# Call-site cheatsheet:
#   F2 vanilla:  pid -> one signal in (B, N). Grid: (B,).
#                BAILEY_EPILOGUE=False, STRIDED_STORE=False.
#                OUTER_DIM and N_TOTAL unused (pass 1 / 0).
#                bt_*_ptr: pass tw_*_ptr again (sentinel; never read).
#   F2-A (F3):   pid -> (b, n1). Grid: (B*N1,). FFT length N=N2.
#                BAILEY_EPILOGUE=True, STRIDED_STORE=False.
#                OUTER_DIM=N1 (n1 = pid % N1).
#                bt_*_ptr: real Bailey twiddles shape (N1, N2).
#   F2-B (F3):   pid -> (b, k2). Grid: (B*N2,). FFT length N=N1.
#                BAILEY_EPILOGUE=False, STRIDED_STORE=True.
#                OUTER_DIM=N2, N_TOTAL=N1*N2.
#                bt_*_ptr: sentinel.

@triton.jit
def f2_kernel(
    x_re_ptr, x_im_ptr,        # (B, N) fp32 input
    y_re_ptr, y_im_ptr,        # (B, N) fp32 output (layout depends on STRIDED_STORE)
    tw_re_ptr, tw_im_ptr,      # (N/2,) fp32 radix-2 twiddles
    perm_ptr,                   # (N,) int32 bit-reversal index
    bt_re_ptr, bt_im_ptr,       # (OUTER_DIM, N) fp32 Bailey twiddles (BAILEY_EPILOGUE only)
    OUTER_DIM, N_TOTAL,
    N: tl.constexpr,
    LOG2_N: tl.constexpr,
    BAILEY_EPILOGUE: tl.constexpr,
    STRIDED_STORE: tl.constexpr,
):
    """Radix-2 Cooley-Tukey FFT in registers, with optional Bailey epilogue and
    strided store. log2(N) butterfly stages via tl.gather for partner shuffle.

    TODO: implement.
    """
    pid = tl.program_id(0)

    offs = tl.arange(0,N)

    perm = tl.load(perm_ptr + offs)
    a_re = tl.load(x_re_ptr + pid * N + perm)
    a_im = tl.load(x_im_ptr + pid * N + perm)

    for s in range(0, LOG2_N):
        half = 1 << s
        step = half << 1

        j = offs & (half - 1)
        base = (offs // step) * step + j

        lo_idx = base
        hi_idx = base + half

        tw_idx = j * (N >> (s + 1))

        w_re = tl.load(tw_re_ptr + tw_idx)
        w_im = tl.load(tw_im_ptr + tw_idx)

        u_re = tl.gather(a_re, lo_idx, 0)
        u_im = tl.gather(a_im, lo_idx, 0)

        v_re = tl.gather(a_re, hi_idx, 0)
        v_im = tl.gather(a_im, hi_idx, 0)

        t_re = w_re * v_re - w_im * v_im
        t_im = w_re * v_im + w_im * v_re

        lo_re = u_re + t_re
        lo_im = u_im + t_im

        hi_re = u_re - t_re
        hi_im = u_im - t_im

        is_lo = (offs & half) == 0

        a_re = tl.where(is_lo, lo_re, hi_re)
        a_im = tl.where(is_lo, lo_im, hi_im)
    
    # for F2-A 
    if BAILEY_EPILOGUE == True:
        outer = pid % OUTER_DIM   
        k2 = offs
        
        # Bailey twiddle 
        bt_re = tl.load(bt_re_ptr + outer * N + k2)
        bt_im = tl.load(bt_im_ptr + outer * N + k2) 

        # temporary re/im multiplicationcd
        tmp_re = a_re * bt_re - a_im * bt_im
        tmp_im = a_re * bt_im + a_im * bt_re

        a_re = tmp_re
        a_im = tmp_im


    # for F2-B
    if STRIDED_STORE == True:
        k2 = pid % OUTER_DIM
        b = pid // OUTER_DIM

        out_offsets = b * N_TOTAL + offs * OUTER_DIM + k2

        tl.store(y_re_ptr + out_offsets, a_re)
        tl.store(y_im_ptr + out_offsets, a_im)
    # if STRIED_STORE == True, then the stores happen here. Otherwise, for vanilla F2 or F2-A, 
    # strided store step is skipped and stores happen below

    else:
        tl.store(y_re_ptr + pid * N + offs, a_re)
        tl.store(y_im_ptr + pid * N + offs, a_im)

def f2_launch(x_re, x_im, y_re, y_im, tw_re, tw_im, perm):
    """Grid: (B,). One program per length-N signal. Vanilla mode.

    TODO: implement.
    """
    B = x_re.shape[0]
    N = x_re.shape[1]
    LOG2_N = int(math.log2(N))

    grid = (B,)

    f2_kernel[grid](
        x_re, x_im,
        y_re, y_im,
        tw_re, tw_im,
        perm,
        tw_re, tw_im,      # sentinel, unused for vanilla F2
        1, 0,              # OUTER_DIM, N_TOTAL unused
        N,
        LOG2_N,
        False,             # BAILEY_EPILOGUE
        False,             # STRIDED_STORE
    )


# =============================================================================
# transpose_kernel: (B, R, C) -> (B, C, R), paired re/im
# =============================================================================

@triton.jit
def transpose_kernel(
    x_re_ptr, x_im_ptr,     # (B*R*C,) fp16 or fp32 input
    y_re_ptr, y_im_ptr,     # (B*R*C,) fp16 or fp32 output
    R, C,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Logical (B, R, C) -> (B, C, R) transpose. Grid: (cdiv(R, BLOCK_R),
    cdiv(C, BLOCK_C), B). Each program copies a (BLOCK_R, BLOCK_C) tile.

    TODO: implement.
    """
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_b = tl.program_id(2)

    r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    # Input index: x[b, r, c] in shape (B, R, C)
    x_offsets = pid_b * R * C + r[:, None] * C + c[None, :]

    # Output index: y[b, c, r] in shape (B, C, R)
    y_offsets = pid_b * R * C + c[None, :] * R + r[:, None]

    mask = (r[:, None] < R) & (c[None, :] < C)

    x_re = tl.load(x_re_ptr + x_offsets, mask=mask, other=0.0)
    x_im = tl.load(x_im_ptr + x_offsets, mask=mask, other=0.0)

    tl.store(y_re_ptr + y_offsets, x_re, mask=mask)
    tl.store(y_im_ptr + y_offsets, x_im, mask=mask)


# =============================================================================
# F4: tcFFT radix-16 single-program FFT (N = 256, L = 2)
# =============================================================================
# See the kernel docstring for the tl.permute tuple-literal gotcha.

@triton.jit
def f4_kernel_L2(
    x_re_ptr, x_im_ptr,    # (B, 256) fp16
    y_re_ptr, y_im_ptr,    # (B, 256) or (B//M, 256, M) fp16
    F_re_ptr, F_im_ptr,    # (16, 16) fp16 -- F_16 DFT matrix
    tw_re_ptr, tw_im_ptr,  # (L=2, 16, 16) fp16 stacked stage twiddles
    B, M,
    BLOCK_B: tl.constexpr,
    STAGE_STOP: tl.constexpr,
    STORE_T: tl.constexpr,
):
    """tcFFT length-256 FFT as two stages of (permute + per-stage twiddle +
    length-16 DFT via four tl.dot). fp16 storage, fp32 matmul accumulators.

    `STAGE_STOP` and `M` are both degenerate in vanilla F4 (`STAGE_STOP=L=2`,
    `M=1`). They exist so the same kernel handles two extra uses:
      - `STAGE_STOP=1`: stop after the s=0 stage, for the sanity_check.py
        stage-1 isolation test (no twiddles, no second matmul).
      - `M>1` with `STORE_T=True`: F7's fused FFT-m_0+T3, writing the
        transposed (rows_outer, 256, M) layout the next level expects.

    STORE_T=False (M=1): natural (B, 256) row-major output.
    STORE_T=True  (M>1): transposed (B//M, 256, M) output for F7 fusion.

    Each stage's four-`tl.dot` is one `_cdot` call; cast its fp32 output to
    fp16 before the next stage.

    Dtype contract:
        Loads:           fp16
        Reshape/permute: fp16 (free)
        tl.dot inputs:   fp16, out_dtype=tl.float32  (use _cdot)
        Twiddle mul:     fp32 * fp16 -> fp32
        Inter-stage:     .to(tl.float16) before next iter's reshape
        Store:           fp16
    Forgetting the inter-stage cast doubles register pressure and passes the
    L=2 tolerance, but fails as soon as F6 stacks more stages.

    Triton 3.6 gotcha -- tl.permute requires LITERAL tuples:
        tl.permute(x, (1, 0, 2))                  # works
        perm = (1, 0, 2); tl.permute(x, perm)     # fails
    Inline each stage's permute tuple at the call site; don't store the
    schedule in a loop variable.

    TODO: implement.
    """
    pid_b = tl.program_id(0)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_n = tl.arange(0, 256)

    mask = offs_b[:, None] < B

    # Load x as (BLOCK_B, 256)
    x_re = tl.load(
        x_re_ptr + offs_b[:, None] * 256 + offs_n[None, :],
        mask=mask,
        other=0.0,
    )

    x_im = tl.load(
        x_im_ptr + offs_b[:, None] * 256 + offs_n[None, :],
        mask=mask,
        other=0.0,
    )

    # Logical shape: (B, d0, d1)
    tile_re = tl.reshape(x_re, (BLOCK_B, 16, 16))
    tile_im = tl.reshape(x_im, (BLOCK_B, 16, 16))

    # Load F^T, where F_T[k, n] = F[n, k]
    kk = tl.arange(0, 16)
    nn = tl.arange(0, 16)

    F_T_re = tl.load(F_re_ptr + nn[None, :] * 16 + kk[:, None])
    F_T_im = tl.load(F_im_ptr + nn[None, :] * 16 + kk[:, None])

# ============================================================
# Stage 0: transform d0 -> e1
# Input layout:  (B, d0, d1)
# Need DFT along d0.
# Move d0 to the last axis for _cdot:
#   (B, d0, d1) -> (B, d1, d0)
# ============================================================

    work_re = tl.permute(tile_re, (0, 2, 1))
    work_im = tl.permute(tile_im, (0, 2, 1))

    mat_re = tl.reshape(work_re, (BLOCK_B * 16, 16))
    mat_im = tl.reshape(work_im, (BLOCK_B * 16, 16))

    mat_re, mat_im = _cdot(mat_re, mat_im, F_T_re, F_T_im)

    mat_re = mat_re.to(tl.float16)
    mat_im = mat_im.to(tl.float16)

# mat is logically (B, d1, e1)
    work_re = tl.reshape(mat_re, (BLOCK_B, 16, 16))
    work_im = tl.reshape(mat_im, (BLOCK_B, 16, 16))

# Put layout as (B, e1, d1), matching the radix-stage bookkeeping
    tile_re = tl.permute(work_re, (0, 2, 1))
    tile_im = tl.permute(work_im, (0, 2, 1))

# ============================================================
# Stage 1: transform d1 -> e0
# Current layout: (B, e1, d1)
# First permute to put d1 in the transform position:
#   (B, e1, d1) -> (B, d1, e1)
# ============================================================

    if STAGE_STOP >= 2:
        tile_re = tl.permute(tile_re, (0, 2, 1))
        tile_im = tl.permute(tile_im, (0, 2, 1))

    # Load stage-1 twiddles tw[1, m, c], where m=d1, c=e1
        m = tl.arange(0, 16)
        c = tl.arange(0, 16)

        tw1_re = tl.load(tw_re_ptr + 1 * 16 * 16 + m[:, None] * 16 + c[None, :])
        tw1_im = tl.load(tw_im_ptr + 1 * 16 * 16 + m[:, None] * 16 + c[None, :])

        tmp_re = tile_re * tw1_re[None, :, :] - tile_im * tw1_im[None, :, :]
        tmp_im = tile_re * tw1_im[None, :, :] + tile_im * tw1_re[None, :, :]

        tile_re = tmp_re.to(tl.float16)
        tile_im = tmp_im.to(tl.float16)

    # Need DFT along d1. Move d1 to last axis for _cdot:
    #   (B, d1, e1) -> (B, e1, d1)
        work_re = tl.permute(tile_re, (0, 2, 1))
        work_im = tl.permute(tile_im, (0, 2, 1))

        mat_re = tl.reshape(work_re, (BLOCK_B * 16, 16))
        mat_im = tl.reshape(work_im, (BLOCK_B * 16, 16))

        mat_re, mat_im = _cdot(mat_re, mat_im, F_T_re, F_T_im)

        mat_re = mat_re.to(tl.float16)
        mat_im = mat_im.to(tl.float16)

    # mat is logically (B, e1, e0)
        work_re = tl.reshape(mat_re, (BLOCK_B, 16, 16))
        work_im = tl.reshape(mat_im, (BLOCK_B, 16, 16))

    # Natural output layout should be (B, e0, e1)
        tile_re = tl.permute(work_re, (0, 2, 1))
        tile_im = tl.permute(work_im, (0, 2, 1))

    if STORE_T:
        row_outer = offs_b // M
        m_idx = offs_b - row_outer * M

        y_offsets = (
            row_outer[:, None] * 256 * M
            + offs_n[None, :] * M
            + m_idx[:, None]
    )
    else:
        y_offsets = offs_b[:, None] * 256 + offs_n[None, :]

    y_re_flat = tl.reshape(tile_re, (BLOCK_B, 256))
    y_im_flat = tl.reshape(tile_im, (BLOCK_B, 256))

    tl.store(y_re_ptr + y_offsets, y_re_flat, mask=mask)
    tl.store(y_im_ptr + y_offsets, y_im_flat, mask=mask)


# =============================================================================
# dft_kernel: padded length-R DFT for the small chunks (R in {2, 4, 8, 16})
# =============================================================================

@triton.jit #used in F6 -- see make_dft_R_padded (twiddles)
def dft_kernel(
    x_re_ptr, x_im_ptr,     # (rows, R) fp16
    y_re_ptr, y_im_ptr,     # (rows, R) or (rows//M, R, M) fp16
    M_re_ptr, M_im_ptr,     # (16, 16) fp16 padded-R DFT matrix
    rows, M,
    R: tl.constexpr,
    BLOCK_B: tl.constexpr,
    STORE_T: tl.constexpr,
):
    """Padded length-R DFT via a (16, 16) tl.dot. STORE_T toggles natural
    vs transposed output (same pattern as f4_kernel_L2).

    One `_cdot(x_re, x_im, MT_re, MT_im)` call replaces the four `tl.dot`
    expansions; cast its fp32 result to fp16 on store.

    TODO: implement.
    """
    pid_b = tl.program_id(0)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_r = tl.arange(0, 16)

    # Load length-R signal padded to length 16.
    x_re = tl.load(
        x_re_ptr + offs_b[:, None] * R + offs_r[None, :],
        mask=(offs_b[:, None] < rows) & (offs_r[None, :] < R),
        other=0.0,
    )

    x_im = tl.load(
        x_im_ptr + offs_b[:, None] * R + offs_r[None, :],
        mask=(offs_b[:, None] < rows) & (offs_r[None, :] < R),
        other=0.0,
    )

    # Load F^T as (16, 16), because _cdot computes x @ F^T.
    k = tl.arange(0, 16)
    n = tl.arange(0, 16)

    M_T_re = tl.load(M_re_ptr + n[None, :] * 16 + k[:, None])
    M_T_im = tl.load(M_im_ptr + n[None, :] * 16 + k[:, None])

    y_re, y_im = _cdot(x_re, x_im, M_T_re, M_T_im)

    # Store only the first R output entries.
    offs_out = tl.arange(0, 16)
    mask_out = (offs_b[:, None] < rows) & (offs_out[None, :] < R)

    if STORE_T:
        # Input rows are logically grouped as:
        #   row = row_outer * M + m_idx
        #
        # Natural output would be:
        #   y[row_outer, m_idx, k]
        #
        # STORE_T=True writes transposed:
        #   y[row_outer, k, m_idx]
        row_outer = offs_b // M
        m_idx = offs_b - row_outer * M

        y_offsets = (
            row_outer[:, None] * R * M
            + offs_out[None, :] * M
            + m_idx[:, None]
        )
    else:
        y_offsets = offs_b[:, None] * R + offs_out[None, :]

    tl.store(
        y_re_ptr + y_offsets,
        y_re.to(tl.float16),
        mask=mask_out,
    )

    tl.store(
        y_im_ptr + y_offsets,
        y_im.to(tl.float16),
        mask=mask_out,
    )


# =============================================================================
# bailey_scale_kernel: elementwise w_N^{n1 kM} multiply with optional fused T2
# =============================================================================

@triton.jit # used in F5?
def bailey_scale_kernel(
    x_re_ptr, x_im_ptr,     # (rows*m0*M,) fp16 input (logical (rows, m0, M))
    y_re_ptr, y_im_ptr,     # (rows*m0*M,) fp16 output ((rows, m0, M) or (rows, M, m0))
    tw_re_ptr, tw_im_ptr,   # (m0, M) fp16
    m0, M,
    BLOCK_M0: tl.constexpr,
    BLOCK_M: tl.constexpr,
    STORE_T: tl.constexpr,
):
    """Elementwise complex multiply by bt[n1, kM] over the (rows, m0, M) view.
    fp32 arithmetic, fp16 result. STORE_T=True fuses with a transpose to
    produce (rows, M, m0).

    Grid: (cdiv(m0, BLOCK_M0), cdiv(M, BLOCK_M), rows).

    TODO: implement.
    """
    pid_m0 = tl.program_id(0)
    pid_M = tl.program_id(1)
    pid_row = tl.program_id(2)

    offs_m0 = pid_m0 * BLOCK_M0 + tl.arange(0, BLOCK_M0)
    offs_M = pid_M * BLOCK_M + tl.arange(0, BLOCK_M)

    mask = (offs_m0[:, None] < m0) & (offs_M[None, :] < M)

    # Input layout: logical (rows, m0, M)
    x_offsets = (
        pid_row * m0 * M
        + offs_m0[:, None] * M
        + offs_M[None, :]
    )

    # Twiddle layout: logical (m0, M)
    tw_offsets = offs_m0[:, None] * M + offs_M[None, :]

    x_re = tl.load(x_re_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    x_im = tl.load(x_im_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)

    tw_re = tl.load(tw_re_ptr + tw_offsets, mask=mask, other=0.0).to(tl.float32)
    tw_im = tl.load(tw_im_ptr + tw_offsets, mask=mask, other=0.0).to(tl.float32)

    y_re = x_re * tw_re - x_im * tw_im
    y_im = x_re * tw_im + x_im * tw_re

    if STORE_T:
        # Output layout: logical (rows, M, m0)
        y_offsets = (
            pid_row * m0 * M
            + offs_M[None, :] * m0
            + offs_m0[:, None]
        )
    else:
        # Output layout: logical (rows, m0, M)
        y_offsets = x_offsets

    tl.store(y_re_ptr + y_offsets, y_re.to(tl.float16), mask=mask)
    tl.store(y_im_ptr + y_offsets, y_im.to(tl.float16), mask=mask)


# =============================================================================
# Thin launch wrappers -- GIVEN, do not edit
# =============================================================================

def _transpose(in_re, in_im, out_re, out_im, B, R, C):
    """Logical (B, R, C) -> (B, C, R) transpose, paired re/im."""
    grid = (triton.cdiv(R, TRANSPOSE_BLOCK), triton.cdiv(C, TRANSPOSE_BLOCK), B)
    transpose_kernel[grid](
        in_re, in_im, out_re, out_im, R, C,
        BLOCK_R=TRANSPOSE_BLOCK, BLOCK_C=TRANSPOSE_BLOCK,
    )

#                                                                   V store_t defaults to False // it's not specified in F6 but in F7, store_t = True in _scale call
def _fft_chunk(in_re, in_im, out_re, out_im, rows, m, plan, M=1, store_t=False):
    """Length-m FFT over `rows` contiguous (rows, m) signals.

    M / store_t control the output layout:
      store_t=False, M=1: natural (rows, m) row-major (F6 leaf path)
      store_t=True,  M>1: transposed (rows//M, m, M) (F7 fused FFT-m0+T3)
    """
    if m == 256:
        f4_plan = plan['f4_plan']
        f4_kernel_L2[(triton.cdiv(rows, F4_L2_BLOCK_B),)](
            in_re.view(rows, 256), in_im.view(rows, 256),
            out_re.view(rows, 256), out_im.view(rows, 256),
            f4_plan['F_re'], f4_plan['F_im'],
            f4_plan['tw_re'], f4_plan['tw_im'],
            rows, M,
            BLOCK_B=F4_L2_BLOCK_B, STAGE_STOP=f4_plan['L'], STORE_T=store_t,
            num_warps=4, num_stages=1,
        )
    else:
        M_re, M_im = plan['dft_mats'][m]
        dft_kernel[(triton.cdiv(rows, DFT_BLOCK_B),)](
            in_re.view(rows, m), in_im.view(rows, m),
            out_re.view(rows, m), out_im.view(rows, m),
            M_re, M_im, rows, M,
            R=m, BLOCK_B=DFT_BLOCK_B, STORE_T=store_t,
        )


def _scale(in_re, in_im, out_re, out_im, rows, m0, M, twr, twi, store_t=False):
    """Bailey scale over logical (rows, m0, M)."""
    grid = (triton.cdiv(m0, SCALE_BLOCK), triton.cdiv(M, SCALE_BLOCK), rows)
    bailey_scale_kernel[grid](
        in_re, in_im, out_re, out_im, twr, twi,
        m0, M, BLOCK_M0=SCALE_BLOCK, BLOCK_M=SCALE_BLOCK, STORE_T=store_t,
    )

    

def _lookup_tw(plan, m0, M, N_i): 
    """Find the precomputed Bailey twiddle table for (m0, M, N_i) in plan['tw']."""
    for (a, b, n, tr, ti) in plan['tw']:
        if a == m0 and b == M and n == N_i:
            return tr, ti
    raise KeyError(f"no twiddle table for (m0={m0}, M={M}, N={N_i})")


# =============================================================================
# F3 pipeline: 4-step Bailey six-step (T1 -> F2-A -> T2 -> F2-B)
# =============================================================================

def f3_launch(in_re, in_im, out_re, out_im, mid_re, mid_im, plan, B):
    """Run the 4-step F3 pipeline. Buffer ping-pong: in -> mid -> out -> mid
    -> out. The Bailey twiddle fuses into F2-A (BAILEY_EPILOGUE=True), and
    the would-be T3 is absorbed by F2-B (STRIDED_STORE=True).

    Steps:
      1. T1 (transpose): x[b, n2, n1] -> A[b, n1, n2]
      2. F2-A:           length-N2 FFT over (B*N1) signals with Bailey epilogue
      3. T2 (transpose): Z[b, n1, k2] -> Z'[b, k2, n1]
      4. F2-B:           length-N1 FFT over (B*N2) signals with strided store

    TODO: implement.
    """
    N  = plan["N"]
    N1 = plan["N1"]
    N2 = plan["N2"] # gave up on making a variable for all of these

    # step 1. T1: view input as (B, N2, N1), transpose to (B, N1, N2)

    _transpose(
        in_re, in_im,
        mid_re, mid_im,
        B, N2, N1,
    )

    # step 2. F2-A: length-N2 FFT over B*N1 signals
    #    Also multiply by Bailey cross twiddle.

    f2_kernel[B * N1,](
        mid_re, mid_im,
        out_re, out_im,
        plan["tw_re_n2"], plan["tw_im_n2"],
        plan["perm_n2"],
        plan["bt_re"], plan["bt_im"],
        N1,          
        N,           
        N2,          
        plan["LOG2_N2"],
        True,        
        False,       
    )

    # step 3. T2: transpose

    _transpose(
        out_re, out_im,
        mid_re, mid_im,
        B, N1, N2,
    )

    # step 4. F2-B: length-N1 FFT over B*N2 signals
    #    Store with stride N2 so final output is laid out as
    #    out[b, k1, k2], i.e. flat index b*N + k1*N2 + k2.

    f2_kernel[B * N2,](
        mid_re, mid_im,
        out_re, out_im,
        plan["tw_re_n1"], plan["tw_im_n1"],
        plan["perm_n1"],
        plan["bt_re"], plan["bt_im"],   
        N2,          
        N,           
        N1,          
        plan["LOG2_N1"],
        False,       
        True,        
    )


# =============================================================================
# F5 pipeline: 6-step Bailey at N1=N2=256 with F4 as inner FFT
# =============================================================================

def f5_launch(in_re, in_im, b0_re, b0_im, b1_re, b1_im, b2_re, b2_im, plan, B):
    """Run the 6-step F5 pipeline at N = 65536 = 256 * 256.

    Buffer ping-pong: in -> b0 -> b1 -> b0 -> b1 -> b2 -> b0 (final).
    The Bailey twiddle is NOT fused into F4 (F4 stays unmodified), so this is
    6 launches; F7 generalizes the fusion idea recursively.

    Steps:
      1. T1:    x[b, n2, n1] -> A[b, n1, n2]
      2. FFT-A: length-256 FFT along last axis -> Y[b, n1, k2]
      3. Scale: Z[b, n1, k2] = Y[b, n1, k2] * bt[n1, k2]
      4. T2:    Z[b, n1, k2] -> Z'[b, k2, n1]
      5. FFT-B: length-256 FFT along last axis -> V[b, k2, k1]
      6. T3:    V[b, k2, k1] -> X[b, k1, k2]   (final in b0)

    TODO: implement.
    """
    N1 = plan["N1"]   # 256
    N2 = plan["N2"]   # 256
    N = plan["N"]     # 65536

    # 1. first transpose x[b, n2, n1] -> A[b,n1,n2]
    _transpose(
        in_re, in_im,
        b0_re, b0_im,
        B, N2, N1,
    )

    # 2. FFT-A: length-256 FFT over B*N1 rows
    _fft_chunk(
        b0_re, b0_im,
        b1_re, b1_im,
        B * N1,
        N2,
        plan,
    )

    # 3. Scale: Z[b, n1, k2] = Y[b, n1, k2] * bt[n1, k2]
    _scale(
        b1_re, b1_im,
        b0_re, b0_im,
        B,
        N1,
        N2,
        plan["bt_re"], plan["bt_im"],
    )

    # 4. T2: Z[b, n1, k2] -> Z'[b, k2, n1]
    _transpose(
        b0_re, b0_im,
        b1_re, b1_im,
        B, N1, N2,
    )

    # 5. FFT-B: length-256 FFT over B*N2 rows
    _fft_chunk(
        b1_re, b1_im,
        b2_re, b2_im,
        B * N2,
        N1,
        plan,
    )

    # 6. T3: V[b, k2, k1] -> X[b, k1, k2], final into b0
    _transpose(
        b2_re, b2_im,
        b0_re, b0_im,
        B, N2, N1,
    )


# =============================================================================
# F6 / F7 recursion
# =============================================================================
# Per level i with chunks = [m_0, m_1, ..., m_{p-1}], M = prod(chunks[1:]):
#   T1 :       (rows, M, m_0) -> (rows, m_0, M)
#   recurse:   length-M FFT over (rows*m_0, M)
#   Scale :    y *= w_{N_i}^{n_1 k_M}            (n_1 = the m_0 digit)
#   T2 :       (rows, m_0, M) -> (rows, M, m_0)
#   FFT-m_0 :  length-m_0 FFT over (rows*M, m_0)
#   T3 :       (rows, M, m_0) -> (rows, m_0, M)   [F6 only; F7 fuses]

def _f6_rec(cur_re, cur_im, rows, chunks, plan, cyc):
    """Recursive 2-factor Bailey split. Leaf (len(chunks)==1) is one
    _fft_chunk call; non-leaf is the 6-step pipeline above.

    Returns the (re, im) cycler-managed buffers holding the (rows, prod(chunks))
    FFT result.

    TODO: implement.
    """
    #================================
    # Case 1: len(chunks) == 1
    if len(chunks) == 1:
        m0 = chunks[0]

        out_re, out_im = cyc.next() # in harness: cyc = _Cycle(bufs, ['a', 'b', 'c']); cyc.next yields next pair in the cycle (I think)

        _fft_chunk(
            cur_re, cur_im,
            out_re, out_im,
            rows,
            m0,
            plan,
        )

        return out_re, out_im
    
    #================================

    # Case 2: 
    m0 = chunks[0]
    M = math.prod(chunks[1:])
    N_i = m0 * M

    # T1
    t1_re, t1_im = cyc.next()

    _transpose(
        cur_re, cur_im,
        t1_re, t1_im,
        rows, M, m0,
    )

    # recurse
    rec_re, rec_im = _f6_rec(
        t1_re, t1_im,
        rows * m0,
        chunks[1:],
        plan,
        cyc,
    )

    # scale
    scaled_re, scaled_im = cyc.next()

    tw_re, tw_im = _lookup_tw(plan, m0, M, N_i)

    _scale(
        rec_re, rec_im,
        scaled_re, scaled_im,
        rows,
        m0,
        M,
        tw_re, tw_im,
        store_t=False
    )

    # T2 
    t2_re, t2_im = cyc.next()

    _transpose(
        scaled_re, scaled_im,
        t2_re, t2_im,
        rows, m0, M,
    )

    # FFT
    fft_re, fft_im = cyc.next()

    _fft_chunk(
        t2_re, t2_im,
        fft_re, fft_im,
        rows * M,
        m0,
        plan,
    )

    # T3 
    out_re, out_im = cyc.next()

    _transpose(
        fft_re, fft_im,
        out_re, out_im,
        rows, M, m0,
    )

    return (out_re, out_im)


def _f7_rec(cur_re, cur_im, rows, chunks, plan, cyc):
    """Same recursion as _f6_rec but with Scale+T2 fused (store_t=True on
    bailey_scale_kernel) and FFT-m_0+T3 fused (store_t=True, M=M on the inner
    FFT kernel). Output should be bitwise-equal to _f6_rec.

    TODO: implement.
    """
    #================================
    # Case 1: len(chunks) == 1
    if len(chunks) == 1:
        m0 = chunks[0]

        out_re, out_im = cyc.next() # in harness: cyc = _Cycle(bufs, ['a', 'b', 'c']); cyc.next yields next pair in the cycle (I think)

        _fft_chunk(
            cur_re, cur_im,
            out_re, out_im,
            rows,
            m0,
            plan,
        )

        return out_re, out_im
    #================================

    # Case 2: 
    m0 = chunks[0]
    M = math.prod(chunks[1:])
    N_i = m0 * M

    # T1 # do not change this transpose
    t1_re, t1_im = cyc.next()

    _transpose(
        cur_re, cur_im,
        t1_re, t1_im,
        rows, M, m0,
    )

    # recurse #CHANGE: to _f7_rec (X)
    rec_re, rec_im = _f7_rec(
        t1_re, t1_im,
        rows * m0,
        chunks[1:],
        plan,
        cyc,
    )

    # scale --- COMBINE WITH T2 (X)

    tw_re, tw_im = _lookup_tw(plan, m0, M, N_i)

    scaled_re, scaled_im = cyc.next()

    _scale(
        rec_re, rec_im,
        scaled_re, scaled_im,
        rows,
        m0,
        M,
        tw_re, tw_im,
        store_t = True
    )

    # T2 # COMBINE WITH SCALE (X)
    #t2_re, t2_im = cyc.next()

    #_transpose(
    #    scaled_re, scaled_im,
    #    t2_re, t2_im,
    #    rows, m0, M,
    #)

    # FFT-m_0 + T3 (X)
    fft_re, fft_im = cyc.next()

    _fft_chunk(
        scaled_re, scaled_im,
        fft_re, fft_im,
        rows * M,
        m0,
        plan,
        M=M,
        store_t = True
    )

    # T3 -- this transpose is absorbed in above step (X)
    #out_re, out_im = cyc.next()

    #_transpose(
    #    fft_re, fft_im,
    #   out_re, out_im,
     #   rows, M, m0,
    #)

    return (fft_re, fft_im)

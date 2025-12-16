import argparse
import time
from typing import Tuple

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from torch import nn

from hy3dgen.shapegen.models.autoencoders.attention_blocks import FourierEmbedder

def build_3d_grid(
    grid_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build a dense 3D grid of coordinates in [0, 1]^3.

    Returns:
        coords: (num_points, 3) tensor
    """
    lin = torch.linspace(0.0, 1.0, grid_size, device=device, dtype=dtype)
    xx, yy, zz = torch.meshgrid(lin, lin, lin, indexing="ij")
    coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    return coords


def build_q_from_grid(
    grid_size: int,
    hidden_size: int,
    num_freqs: int = 6,
    logspace: bool = True,
    include_input: bool = True,
    include_pi: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, nn.Module]:
    """
    Build Q matrix using Hunyuan's Fourier positional encoding and a linear projection,
    mimicking the query path of the cross-attention decoder.

    Args:
        grid_size: number of grid points per axis (total points = grid_size^3)
        hidden_size: width of the cross-attention (H)
        num_freqs, logspace, include_input, include_pi: FourierEmbedder params

    Returns:
        Q: (1, num_points, hidden_size) tensor
        query_proj: the nn.Linear used to project positional encodings to Q
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Positional encoding as in Hunyuan (FourierEmbedder)
    fourier_embedder = FourierEmbedder(
        num_freqs=num_freqs,
        logspace=logspace,
        input_dim=3,
        include_input=include_input,
        include_pi=include_pi,
    ).to(device)

    coords = build_3d_grid(grid_size, device=device, dtype=dtype)  # (P, 3)
    pe = fourier_embedder(coords)  # (P, pe_dim)

    query_proj = nn.Linear(fourier_embedder.out_dim, hidden_size, bias=True).to(device, dtype)

    Q = query_proj(pe)  # (P, H)
    Q = Q.unsqueeze(0)  # add batch dim -> (1, P, H)
    return Q, query_proj, fourier_embedder


def build_random_k(
    batch_size: int,
    num_keys: int,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build a random K matrix compatible with Q for matmul.

    Returns:
        K: (batch_size, num_keys, hidden_size)
    """
    return torch.randn(batch_size, num_keys, hidden_size, device=device, dtype=dtype)


def fast_qk_matmul(
    query_proj: nn.Module,
    fourier_embedder: nn.Module,
    K: torch.Tensor,
    grid_size: int,
) -> torch.Tensor:
    """
    Compute Q @ K^T efficiently by exploiting the additive structure of the Fourier embedding.
    Q(x,y,z) = Q_x(x) + Q_y(y) + Q_z(z) + bias
    S(x,y,z, k) = <Q(x,y,z), K_k> = S_x(x,k) + S_y(y,k) + S_z(z,k) + S_b(k)

    Args:
        query_proj: Linear layer projecting embeddings to H
        fourier_embedder: The embedding module
        K: (B, Nk, H)
        grid_size: G

    Returns:
        S: (B, Nq, Nk) where Nq = G^3. Flattened to match naive output.
    """
    device = K.device
    dtype = K.dtype
    B, Nk, H = K.shape

    # 1. Extract weights and separate them by axis
    # W: (H, D_in), b: (H,)
    W = query_proj.weight.to(dtype)  # (H, InDim)
    b = query_proj.bias.to(dtype) if query_proj.bias is not None else torch.zeros(H, device=device, dtype=dtype)

    num_freqs = fourier_embedder.num_freqs
    include_input = fourier_embedder.include_input
    
    # Calculate indices for each axis
    # Structure: [x, y, z, sin(fx)..., sin(fy)..., sin(fz)..., cos(fx)..., cos(fy)..., cos(fz)...]
    # Input part: 0, 1, 2
    # Sin part: 3 blocks of num_freqs
    # Cos part: 3 blocks of num_freqs
    
    sin_start = 3 if include_input else 0
    cos_start = sin_start + 3 * num_freqs
    
    indices = [[], [], []] # x, y, z indices
    
    if include_input:
        indices[0].append(0)
        indices[1].append(1)
        indices[2].append(2)
        
    for axis in range(3):
        # Sin indices for this axis
        start = sin_start + axis * num_freqs
        end = start + num_freqs
        indices[axis].extend(range(start, end))
        
        # Cos indices for this axis
        start = cos_start + axis * num_freqs
        end = start + num_freqs
        indices[axis].extend(range(start, end))
        
    # 2. Precompute 1D coordinate embeddings
    # We need to apply the embedding to a 1D line and project it using the sliced weights.
    # Actually, simpler: compute full embedding of a line like (t, 0, 0), extract relevant columns, multiply by full W?
    # No, that mixes biases and other 0-terms.
    # Better: Construct the sub-embedding matrix for 1D coords and multiply by sub-matrix of W.
    
    lin = torch.linspace(0.0, 1.0, grid_size, device=device, dtype=dtype) # (G,)
    
    # Calculate embedding components for the 1D line
    # PE(t) = [t, sin(f*t), cos(f*t)] (conceptually, for the relevant indices)
    
    # We can use the embedder's frequencies
    freqs = fourier_embedder.frequencies.to(dtype) # (F,)
    
    # (G, F)
    angles = lin[:, None] * freqs[None, :] 
    emb_sin = torch.sin(angles)
    emb_cos = torch.cos(angles)
    
    # Construct feature matrices for each axis: (G, Dim_per_axis)
    # Each axis has: [coord] + [sin terms] + [cos terms] (depending on include_input)
    # We stack them to form a (G, D_axis) matrix
    
    feats = []
    if include_input:
        feats.append(lin[:, None]) # (G, 1)
    feats.append(emb_sin)
    feats.append(emb_cos)
    
    # This 'feats' corresponds to the columns selected by indices[axis] for ANY axis,
    # because the embedding function is identical for x, y, z.
    phi_1d = torch.cat(feats, dim=1) # (G, D_axis)
    
    # 3. Compute Projected components: Q_axis = phi_1d @ W_axis.T
    # W_axis is (H, D_axis)
    
    qs = []
    for axis in range(3):
        w_idx = indices[axis]
        W_sub = W[:, w_idx] # (H, D_axis)
        # Q_sub: (G, H)
        Q_sub = phi_1d @ W_sub.T 
        qs.append(Q_sub)
        
    Q_x, Q_y, Q_z = qs
    
    # 4. Compute Scores components: S_axis = Q_axis @ K.T
    # K: (B, Nk, H)
    # Q_sub: (G, H) -> broadcast to (B, G, H) if needed, or just matmul
    
    # We want S_axis: (B, G, Nk)
    # Q_sub is (G, H). K is (B, Nk, H).
    # Q_sub @ K.transpose -> (G, H) @ (B, H, Nk) -> (B, G, Nk)
    
    # (B, Nk, H) -> (B, H, Nk)
    Kt = K.transpose(-2, -1)
    
    S_x = torch.matmul(Q_x, Kt) # (B, G, Nk)
    S_y = torch.matmul(Q_y, Kt)
    S_z = torch.matmul(Q_z, Kt)
    
    # Bias term: b @ K.T -> (H,) @ (B, H, Nk) -> (B, Nk)
    # reshape to (B, 1, 1, 1, Nk) for broadcasting
    S_b = torch.matmul(b, Kt) # (B, Nk)
    
    # 5. Sum and Broadcast
    # Target shape: (B, Nq, Nk) where Nq = G*G*G.
    # Grid order is (x, y, z) with z fastest (from meshgrid ij + reshape)
    # So flattened index maps to x * G^2 + y * G + z.
    # This corresponds to tensor shape (G, G, G).
    # S_x corresponds to dim 0. S_y to dim 1. S_z to dim 2.
    
    # Reshape for broadcasting:
    # S_x: (B, G, 1, 1, Nk)
    # S_y: (B, 1, G, 1, Nk)
    # S_z: (B, 1, 1, G, Nk)
    
    S_x_bcd = S_x.unsqueeze(2).unsqueeze(2) # (B, G, 1, 1, Nk)
    S_y_bcd = S_y.unsqueeze(1).unsqueeze(3) # (B, 1, G, 1, Nk)
    S_z_bcd = S_z.unsqueeze(1).unsqueeze(1) # (B, 1, 1, G, Nk)
    S_b_bcd = S_b.view(B, 1, 1, 1, Nk)
    
    S_vol = S_x_bcd + S_y_bcd + S_z_bcd + S_b_bcd # (B, G, G, G, Nk)
    
    return S_vol.reshape(B, -1, Nk)


def fast_qk_matmul_factored(
    query_proj: nn.Module,
    fourier_embedder: nn.Module,
    K: torch.Tensor,
    grid_size: int,
) -> torch.Tensor:
    """
    Faster variant of fast_qk_matmul using reassociation to remove the H dimension from the per-axis matmuls.

    Let (for one axis) Q_x = Phi_x @ W_x^T, where:
      - Phi_x: (G, D) are Fourier features for 1D coordinates
      - W_x:   (H, D) are the corresponding columns of query_proj.weight

    Then the per-axis score contribution is:
        S_x = Q_x @ K^T = (Phi_x @ W_x^T) @ K^T = Phi_x @ (W_x^T @ K^T)

    Here (W_x^T @ K^T) has shape (D, Nk) (per batch), with D << H, so this is much cheaper than forming Q_x (G, H)
    and multiplying by K^T (H, Nk).

    Returns:
        S: (B, G^3, Nk)
    """
    device = K.device
    dtype = K.dtype
    B, Nk, H = K.shape

    W = query_proj.weight.to(dtype)  # (H, InDim)
    b = query_proj.bias.to(dtype) if query_proj.bias is not None else torch.zeros(H, device=device, dtype=dtype)

    num_freqs = fourier_embedder.num_freqs
    include_input = fourier_embedder.include_input

    sin_start = 3 if include_input else 0
    cos_start = sin_start + 3 * num_freqs

    indices = [[], [], []]  # x, y, z indices
    if include_input:
        indices[0].append(0)
        indices[1].append(1)
        indices[2].append(2)

    for axis in range(3):
        start = sin_start + axis * num_freqs
        indices[axis].extend(range(start, start + num_freqs))
        start = cos_start + axis * num_freqs
        indices[axis].extend(range(start, start + num_freqs))

    # Build 1D Fourier features Phi(t) for t in linspace(0, 1, G)
    lin = torch.linspace(0.0, 1.0, grid_size, device=device, dtype=dtype)  # (G,)
    freqs = fourier_embedder.frequencies.to(dtype)  # (F,)
    angles = lin[:, None] * freqs[None, :]  # (G, F)
    emb_sin = torch.sin(angles)
    emb_cos = torch.cos(angles)

    feats = []
    if include_input:
        feats.append(lin[:, None])  # (G, 1)
    feats.append(emb_sin)  # (G, F)
    feats.append(emb_cos)  # (G, F)
    phi_1d = torch.cat(feats, dim=1)  # (G, D_axis)

    Kt = K.transpose(-2, -1)  # (B, H, Nk)

    # Bias contribution: (H,) @ (B, H, Nk) -> (B, Nk)
    S_b = torch.matmul(b, Kt).view(B, 1, 1, 1, Nk)

    # For each axis: compute M_axis = W_sub^T @ K^T (B, D, Nk), then S_axis = Phi @ M_axis (B, G, Nk)
    S_axes = []
    for axis in range(3):
        w_idx = indices[axis]
        W_sub = W[:, w_idx]  # (H, D)
        # (D, H) @ (B, H, Nk) -> (B, D, Nk)
        M_axis = torch.matmul(W_sub.transpose(0, 1), Kt)
        # (G, D) @ (B, D, Nk) -> (B, G, Nk)
        S_axis = torch.matmul(phi_1d, M_axis)
        S_axes.append(S_axis)

    S_x, S_y, S_z = S_axes

    # Broadcast sum to (B, G, G, G, Nk), then flatten to (B, G^3, Nk)
    S_vol = (
        S_x.unsqueeze(2).unsqueeze(2)  # (B, G, 1, 1, Nk)
        + S_y.unsqueeze(1).unsqueeze(3)  # (B, 1, G, 1, Nk)
        + S_z.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, G, Nk)
        + S_b
    )
    return S_vol.reshape(B, -1, Nk)


def fast_qk_matmul_fft(
    query_proj: nn.Module,
    fourier_embedder: nn.Module,
    K: torch.Tensor,
    grid_size: int,
) -> torch.Tensor:
    """
    FFT-accelerated variant using FFT for efficient 3D outer product computation.
    
    The key optimization: Instead of broadcasting S_x, S_y, S_z and summing, we use FFT
    to compute the 3D tensor product more efficiently. For large grids, FFT-based
    convolution/outer product can be faster than explicit broadcasting.
    
    Additionally, we optimize the matmul phi_1d @ M_axis by using FFT when beneficial.
    
    Returns:
        S: (B, G^3, Nk)
    """
    device = K.device
    dtype = K.dtype
    B, Nk, H = K.shape

    W = query_proj.weight.to(dtype)  # (H, InDim)
    b = query_proj.bias.to(dtype) if query_proj.bias is not None else torch.zeros(H, device=device, dtype=dtype)

    num_freqs = fourier_embedder.num_freqs
    include_input = fourier_embedder.include_input

    sin_start = 3 if include_input else 0
    cos_start = sin_start + 3 * num_freqs

    indices = [[], [], []]  # x, y, z indices
    if include_input:
        indices[0].append(0)
        indices[1].append(1)
        indices[2].append(2)

    for axis in range(3):
        start = sin_start + axis * num_freqs
        indices[axis].extend(range(start, start + num_freqs))
        start = cos_start + axis * num_freqs
        indices[axis].extend(range(start, start + num_freqs))

    # Build 1D Fourier features Phi(t) for t in linspace(0, 1, G)
    lin = torch.linspace(0.0, 1.0, grid_size, device=device, dtype=dtype)  # (G,)
    freqs = fourier_embedder.frequencies.to(dtype)  # (F,)
    angles = lin[:, None] * freqs[None, :]  # (G, F)
    emb_sin = torch.sin(angles)
    emb_cos = torch.cos(angles)

    feats = []
    if include_input:
        feats.append(lin[:, None])  # (G, 1)
    feats.append(emb_sin)  # (G, F)
    feats.append(emb_cos)  # (G, F)
    phi_1d = torch.cat(feats, dim=1)  # (G, D_axis)

    Kt = K.transpose(-2, -1)  # (B, H, Nk)

    # Bias contribution: (H,) @ (B, H, Nk) -> (B, Nk)
    S_b = torch.matmul(b, Kt)  # (B, Nk)

    # For each axis: compute M_axis = W_sub^T @ K^T (B, D, Nk), then S_axis = Phi @ M_axis (B, G, Nk)
    # Use FFT-based optimization: compute matmul in frequency domain when beneficial
    S_axes = []
    D_axis = phi_1d.shape[1]
    
    # Use FFT for matmul when grid is large enough to benefit
    use_fft = grid_size >= 64
    
    for axis in range(3):
        w_idx = indices[axis]
        W_sub = W[:, w_idx]  # (H, D)
        # (D, H) @ (B, H, Nk) -> (B, D, Nk)
        M_axis = torch.matmul(W_sub.transpose(0, 1), Kt)  # (B, D, Nk)
        
        if use_fft and D_axis >= 8:
            # FFT-based matmul: Express phi_1d @ M_axis using FFT convolution
            # For each (b, k), we compute: sum_d phi_1d[:, d] * M_axis[b, d, k]
            # This can be computed using FFT by treating it as a weighted sum of signals
            
            # Method: Use FFT to compute the weighted combination
            # phi_1d: (G, D) - each column is a 1D signal
            # M_axis: (B, D, Nk) - weights for each signal
            
            # Compute FFT of each column of phi_1d
            # phi_1d_fft: (G//2+1, D) in frequency domain
            phi_1d_fft = torch.fft.rfft(phi_1d, dim=0)  # (G//2+1, D)
            
            # For each (b, k), compute weighted sum in frequency domain
            # Result: (B, G//2+1, Nk) = einsum('gd,bdk->bgk', phi_1d_fft, M_axis)
            # But we need to handle complex numbers and batch dimension
            
            # Reshape for broadcasting: phi_1d_fft (G//2+1, D), M_axis (B, D, Nk)
            # We want: sum_d phi_1d_fft[:, d] * M_axis[b, d, k] for each (b, k)
            # This is: (G//2+1, D) @ (B, D, Nk) -> (B, G//2+1, Nk)
            
            # Use einsum for clarity: 'gd,bdk->bgk'
            S_axis_fft = torch.einsum('gd,bdk->bgk', phi_1d_fft, M_axis)  # (B, G//2+1, Nk)
            
            # Transform back to spatial domain
            S_axis = torch.fft.irfft(S_axis_fft, n=grid_size, dim=1)  # (B, G, Nk)
        else:
            # Standard matmul (efficient for small D or small G)
            S_axis = torch.matmul(phi_1d, M_axis)  # (B, G, Nk)
        
        S_axes.append(S_axis)

    S_x, S_y, S_z = S_axes  # Each: (B, G, Nk)

    # Optimized 3D outer product with better memory access
    # Match the broadcasting pattern from fast_qk_matmul_factored exactly
    S_vol = (
        S_x.unsqueeze(2).unsqueeze(2)  # (B, G, 1, 1, Nk)
        + S_y.unsqueeze(1).unsqueeze(3)  # (B, 1, G, 1, Nk)
        + S_z.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, G, Nk)
        + S_b.view(B, 1, 1, 1, Nk)
    )
    
    return S_vol.reshape(B, -1, Nk)


def fast_qk_matmul_cooley_tukey(
    query_proj: nn.Module,
    fourier_embedder: nn.Module,
    K: torch.Tensor,
    grid_size: int,
) -> torch.Tensor:
    """
    Practical “Cooley–Tukey-style” optimization: **fuse work and reduce kernel launches**.

    Important: a true Cooley–Tukey FFT speedup only applies when you're multiplying by a
    DFT matrix (integer Fourier modes on a uniform grid). Here, `FourierEmbedder` uses a
    small set of sin/cos features (typically 6 freqs) and the dominant cost is the
    mixing over keys \(Nk\). Doing FFTs on `phi_1d` does not reduce the big-O term.

    What *does* help in practice on GPU is:
    - compute the 3 per-axis projections in a **single GEMM** (better utilization)
    - compute the 3 per-axis score lines via **batched GEMM** (fewer launches)

    This keeps the math identical to `fast_qk_matmul_factored`, but is often faster.

    Returns:
        S: (B, G^3, Nk)
    """
    device = K.device
    dtype = K.dtype
    B, Nk, H = K.shape

    W = query_proj.weight.to(dtype)  # (H, InDim)
    b = (
        query_proj.bias.to(dtype)
        if query_proj.bias is not None
        else torch.zeros(H, device=device, dtype=dtype)
    )

    num_freqs = fourier_embedder.num_freqs
    include_input = fourier_embedder.include_input

    sin_start = 3 if include_input else 0
    cos_start = sin_start + 3 * num_freqs

    # Indices into FourierEmbedder output for each axis
    idx_x: list[int] = []
    idx_y: list[int] = []
    idx_z: list[int] = []
    if include_input:
        idx_x.append(0)
        idx_y.append(1)
        idx_z.append(2)

    for axis, idx in enumerate((idx_x, idx_y, idx_z)):
        start = sin_start + axis * num_freqs
        idx.extend(range(start, start + num_freqs))
        start = cos_start + axis * num_freqs
        idx.extend(range(start, start + num_freqs))

    # Build 1D Fourier features Phi(t) for t in linspace(0, 1, G)
    lin = torch.linspace(0.0, 1.0, grid_size, device=device, dtype=dtype)  # (G,)
    freqs = fourier_embedder.frequencies.to(dtype)  # (F,)
    angles = lin[:, None] * freqs[None, :]  # (G, F)
    emb_sin = torch.sin(angles)
    emb_cos = torch.cos(angles)

    feats = []
    if include_input:
        feats.append(lin[:, None])  # (G, 1)
    feats.append(emb_sin)  # (G, F)
    feats.append(emb_cos)  # (G, F)
    phi_1d = torch.cat(feats, dim=1)  # (G, D)
    D = phi_1d.shape[1]

    # Bias contribution: b @ K^T  ==  K @ b
    # K: (B, Nk, H), b: (H,) -> (B, Nk)
    S_b = torch.matmul(K, b)  # (B, Nk)

    # ---- Fused projections (single GEMM instead of 3) ----
    # W_cat: (H, 3D)
    W_cat = torch.cat([W[:, idx_x], W[:, idx_y], W[:, idx_z]], dim=1)
    # K_proj_cat: (B, Nk, 3D)
    K_proj_cat = torch.matmul(K, W_cat)
    # Reshape into three blocks: (B, 3, Nk, D)
    K_proj_3 = K_proj_cat.view(B, Nk, 3, D).permute(0, 2, 1, 3).contiguous()

    # ---- Batched GEMM for the 3 axes (single bmm over B*3) ----
    # Want S_axis = phi_1d (G, D) @ (K_proj_axis^T) (D, Nk) -> (G, Nk)
    # Prepare: M_stack = (B*3, D, Nk)
    M_stack = K_proj_3.permute(0, 1, 3, 2).reshape(B * 3, D, Nk).contiguous()
    # Prepare: Phi_stack = (B*3, G, D)
    Phi_stack = phi_1d.unsqueeze(0).expand(B * 3, -1, -1).contiguous()
    # (B*3, G, Nk)
    S_stack = torch.bmm(Phi_stack, M_stack)
    # (B, 3, G, Nk)
    S_stack = S_stack.view(B, 3, grid_size, Nk)
    S_x = S_stack[:, 0]
    S_y = S_stack[:, 1]
    S_z = S_stack[:, 2]

    # Broadcast sum to (B, G, G, G, Nk), then flatten to (B, G^3, Nk)
    S_vol = (
        S_x.unsqueeze(2).unsqueeze(2)  # (B, G, 1, 1, Nk)
        + S_y.unsqueeze(1).unsqueeze(3)  # (B, 1, G, 1, Nk)
        + S_z.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, G, Nk)
        + S_b.view(B, 1, 1, 1, Nk)
    )

    return S_vol.reshape(B, -1, Nk)


def build_random_v(
    batch_size: int,
    num_keys: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.randn(batch_size, num_keys, head_dim, device=device, dtype=dtype)


def get_block_reordering_perm(grid_size: int, block_size: int = 8, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Generate a permutation index to reorder a flattened 3D grid (x, y, z) 
    into a block-wise order to improve spatial locality.
    """
    assert grid_size % block_size == 0, "Grid size must be divisible by block size"
    num_blocks = grid_size // block_size
    
    # Original indices: (G, G, G)
    # We want to visit blocks (bx, by, bz) of size (bs, bs, bs)
    # Dimensions: (Nb, Bs, Nb, Bs, Nb, Bs)
    
    # 1. Create linear indices
    indices = torch.arange(grid_size**3, device=device).view(grid_size, grid_size, grid_size)
    
    # 2. Reshape to separate blocks
    # (Nb, Bs, Nb, Bs, Nb, Bs)
    indices = indices.view(num_blocks, block_size, num_blocks, block_size, num_blocks, block_size)
    
    # 3. Permute to bring blocks together: (Nb, Nb, Nb, Bs, Bs, Bs)
    indices = indices.permute(0, 2, 4, 1, 3, 5).contiguous()
    
    # 4. Flatten
    perm = indices.view(-1)
    return perm


def get_col_centroid_perm(sparse_matrix_csr: torch.Tensor) -> torch.Tensor:
    """
    Reorder columns based on the centroid of their row indices.
    For each column j, centroid(j) = mean(row_indices where M[i, j] != 0).
    We sort columns by this centroid.
    Assumes sparse_matrix_csr is 2D (batch dims handled separately or averaged).
    """
    # Convert to COO to get indices easily
    coo = sparse_matrix_csr.to_sparse_coo()
    rows = coo.indices()[0].float()
    cols = coo.indices()[1]
    
    num_cols = sparse_matrix_csr.shape[1]
    
    # Sum of row indices for each col
    row_sum = torch.zeros(num_cols, device=sparse_matrix_csr.device, dtype=torch.float32)
    row_sum.scatter_add_(0, cols, rows)
    
    # Count non-zeros per column
    col_counts = torch.zeros(num_cols, device=sparse_matrix_csr.device, dtype=torch.float32)
    col_counts.scatter_add_(0, cols, torch.ones_like(rows, dtype=torch.float32))
    
    centroids = row_sum / (col_counts + 1e-6)
    
    # Sort columns by centroid
    perm = torch.argsort(centroids)
    return perm


def benchmark_sparse_matmul(
    S_sparse: torch.Tensor,
    V: torch.Tensor,
    warmup: int = 5,
    iters: int = 20,
) -> float:
    """
    Benchmark S_sparse @ V
    """
    device = S_sparse.device
    
    def run():
        return torch.mm(S_sparse, V)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    for _ in range(warmup):
        _ = run()
        
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    start = time.perf_counter()
    for _ in range(iters):
        _ = run()
        
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    
    return ((end - start) / iters) * 1e3


def benchmark_qk_matmul(
    Q: torch.Tensor,
    K: torch.Tensor,
    warmup: int = 5,
    iters: int = 20,
) -> float:
    """
    Benchmark Q @ K^T using torch.matmul.

    Args:
        Q: (B, Nq, H)
        K: (B, Nk, H)

    Returns:
        average_time_ms: average wall-clock time per iteration in milliseconds
    """
    assert Q.shape[0] == K.shape[0], "Batch dimensions of Q and K must match"
    assert Q.shape[2] == K.shape[2], "Last dimension (H) of Q and K must match"

    device = Q.device
    use_cuda = device.type == "cuda"

    def _sync():
        if use_cuda:
            torch.cuda.synchronize(device)

    # Warmup
    for _ in range(warmup):
        _ = torch.matmul(Q, K.transpose(-2, -1))
    _sync()

    # Timed runs
    start = time.perf_counter()
    for _ in range(iters):
        _ = torch.matmul(Q, K.transpose(-2, -1))
    _sync()
    end = time.perf_counter()

    total_time = end - start
    avg_time_ms = (total_time / iters) * 1e3
    return avg_time_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark QK matmul using Hunyuan-style cross-attention shapes."
    )
    parser.add_argument("--grid-size", type=int, default=32, help="Grid size per axis (total points = G^3).")
    parser.add_argument("--hidden-size", type=int, default=1024, help="Hidden size H for Q and K.")
    parser.add_argument("--num-keys", type=int, default=4096, help="Number of key positions (Nk).")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (B).")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations before timing.")
    parser.add_argument("--iters", type=int, default=50, help="Number of timed iterations.")
    return parser.parse_args()


def str_to_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = str_to_dtype(args.dtype)
    print(f"Device: {device}")
    print(f"dtype: {dtype}")

    Q, query_proj, fourier_embedder = build_q_from_grid(
        grid_size=args.grid_size,
        hidden_size=args.hidden_size,
        device=device,
        dtype=dtype,
    )

    K = build_random_k(
        batch_size=args.batch_size,
        num_keys=args.num_keys,
        hidden_size=args.hidden_size,
        device=device,
        dtype=dtype,
    )

    # If batch_size > 1, tile Q over the batch dimension
    if args.batch_size > 1:
        Q_expanded = Q.expand(args.batch_size, -1, -1).contiguous()
    else:
        Q_expanded = Q

    print(f"Device: {device}")
    print(f"dtype: {dtype}")
    print(f"Q shape: {tuple(Q_expanded.shape)}  (B, Nq, H)")
    print(f"K shape: {tuple(K.shape)}  (B, Nk, H)")

    # 1. Correctness Check
    print("Checking correctness...")
    S_naive = torch.matmul(Q_expanded, K.transpose(-2, -1))
    S_fast = fast_qk_matmul(query_proj, fourier_embedder, K, args.grid_size)
    S_fast2 = fast_qk_matmul_factored(query_proj, fourier_embedder, K, args.grid_size)
    # S_fft = fast_qk_matmul_fft(query_proj, fourier_embedder, K, args.grid_size)
    S_cooley_tukey = fast_qk_matmul_cooley_tukey(query_proj, fourier_embedder, K, args.grid_size)
    
    diff = (S_naive - S_fast).abs().max()
    print(f"Max difference between naive and fast: {diff.item():.6e}")
    if diff > 1e-2:
        print("WARNING: Difference is large!")
    else:
        print("Correctness verified.")

    diff2 = (S_fast - S_fast2).abs().max()
    print(f"Max difference between fast and fast2(factored): {diff2.item():.6e}")
    
    # diff3 = (S_naive - S_fft).abs().max()
    # print(f"Max difference between naive and FFT: {diff3.item():.6e}")
    # if diff3 > 1e-2:
    #     print("WARNING: FFT difference is large!")
    
    diff4 = (S_naive - S_cooley_tukey).abs().max()
    print(f"Max difference between naive and Cooley-Tukey: {diff4.item():.6e}")
    if diff4 > 1e-2:
        print("WARNING: Cooley-Tukey difference is large!")

    # 2. Benchmark Naive
    print(f"\nBenchmarking Naive Implementation...")
    avg_time_naive = benchmark_qk_matmul(Q_expanded, K, warmup=args.warmup, iters=args.iters)
    print(f"Naive Time: {avg_time_naive:.3f} ms")

    # 3. Benchmark Fast
    print(f"\nBenchmarking Fast Implementation...")
    # Wrap in lambda to match signature expected? No, benchmark_qk_matmul expects Q, K. 
    # We need a custom timing loop for fast implementation since it has different signature.
    
    def run_fast():
        return fast_qk_matmul(query_proj, fourier_embedder, K, args.grid_size)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    for _ in range(args.warmup):
        _ = run_fast()
        
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    start = time.perf_counter()
    for _ in range(args.iters):
        _ = run_fast()
        
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_time_fast = ((end - start) / args.iters) * 1e3
    print(f"Fast Time:  {avg_time_fast:.3f} ms")
    print(f"Speedup:    {avg_time_naive / avg_time_fast:.2f}x")

    # 3b. Benchmark Fast2 (Factored)
    print(f"\nBenchmarking Fast2 (Factored) Implementation...")

    def run_fast2():
        return fast_qk_matmul_factored(query_proj, fourier_embedder, K, args.grid_size)

    if device.type == "cuda":
        torch.cuda.synchronize()

    for _ in range(args.warmup):
        _ = run_fast2()

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(args.iters):
        _ = run_fast2()

    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time_fast2 = ((end - start) / args.iters) * 1e3
    print(f"Fast2 Time: {avg_time_fast2:.3f} ms")
    print(f"Speedup vs naive: {avg_time_naive / avg_time_fast2:.2f}x")
    print(f"Speedup vs fast:  {avg_time_fast / avg_time_fast2:.2f}x")

    # 3c. Benchmark FFT Implementation
    # print(f"\nBenchmarking FFT Implementation...")

    # def run_fft():
    #     return fast_qk_matmul_fft(query_proj, fourier_embedder, K, args.grid_size)

    # if device.type == "cuda":
    #     torch.cuda.synchronize()

    # for _ in range(args.warmup):
    #     _ = run_fft()

    # if device.type == "cuda":
    #     torch.cuda.synchronize()

    # start = time.perf_counter()
    # for _ in range(args.iters):
    #     _ = run_fft()

    # if device.type == "cuda":
    #     torch.cuda.synchronize()
    # end = time.perf_counter()

    # avg_time_fft = ((end - start) / args.iters) * 1e3
    # print(f"FFT Time:   {avg_time_fft:.3f} ms")
    # print(f"Speedup vs naive: {avg_time_naive / avg_time_fft:.2f}x")
    # print(f"Speedup vs fast:  {avg_time_fast / avg_time_fft:.2f}x")
    # print(f"Speedup vs fast2: {avg_time_fast2 / avg_time_fft:.2f}x")

    # 3d. Benchmark Cooley-Tukey FFT Implementation
    print(f"\nBenchmarking Cooley-Tukey FFT Implementation...")

    def run_cooley_tukey():
        return fast_qk_matmul_cooley_tukey(query_proj, fourier_embedder, K, args.grid_size)

    if device.type == "cuda":
        torch.cuda.synchronize()

    for _ in range(args.warmup):
        _ = run_cooley_tukey()

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(args.iters):
        _ = run_cooley_tukey()

    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time_cooley_tukey = ((end - start) / args.iters) * 1e3
    print(f"Cooley-Tukey Time: {avg_time_cooley_tukey:.3f} ms")
    print(f"Speedup vs naive: {avg_time_naive / avg_time_cooley_tukey:.2f}x")
    print(f"Speedup vs fast:  {avg_time_fast / avg_time_cooley_tukey:.2f}x")
    print(f"Speedup vs fast2: {avg_time_fast2 / avg_time_cooley_tukey:.2f}x")
    # NOTE: FFT benchmark may be disabled; don't assume avg_time_fft exists.

    # 4. Sparse Benchmark with Reordering
    print(f"\nBenchmarking Sparse Reordering...")
    
    # Create V matrix
    V = build_random_v(args.batch_size, args.num_keys, args.hidden_size, device, dtype).squeeze(0) # Assume B=1
    
    # Create sparse S (simulating softmax + topk/threshold)
    # Use the fastest dense score tensor. Squeeze B.
    S_dense = S_fast2.squeeze(0).float()  # Sparse ops often prefer float32 indices/values or specific types
    
    # Keep top 1% values
    k = int(args.num_keys * 0.01)
    if k == 0: k = 1
    
    # Top-k per row
    vals, indices = torch.topk(S_dense, k, dim=1)
    
    # Construct CSR
    rows = torch.arange(S_dense.shape[0], device=device).repeat_interleave(k)
    cols = indices.flatten()
    data = vals.flatten()
    
    S_sparse = torch.sparse_csr_tensor(
        torch.arange(0, len(data) + 1, k, device=device), # crow_indices
        cols,
        data,
        size=S_dense.shape,
        dtype=dtype
    )
    
    # Benchmark Baseline Sparse
    time_sparse_base = benchmark_sparse_matmul(S_sparse, V, args.warmup, args.iters)
    print(f"Sparse Matmul (Natural Order): {time_sparse_base:.3f} ms")
    
    # Reordering
    # 1. Row Reordering (Block 3D)
    block_size = 8
    perm_r = get_block_reordering_perm(args.grid_size, block_size, device)
    
    # Apply row permutation: P @ S
    # For CSR, this effectively shuffles the rows.
    # We can reconstruct easily.
    # To do it efficiently on existing sparse tensor is tricky, better to rebuild.
    
    # Re-gather from dense for simplicity of implementation (in real scenario, we would map indices)
    # S_reordered_rows = S_dense[perm_r] -> TopK
    
    # Or just index the dense S (since we have it) to create reordered sparse S
    S_dense_reordered_r = S_dense[perm_r]
    vals_r, indices_r = torch.topk(S_dense_reordered_r, k, dim=1)
    
    rows_r = torch.arange(S_dense.shape[0], device=device).repeat_interleave(k)
    cols_r = indices_r.flatten()
    data_r = vals_r.flatten()
    
    S_sparse_r = torch.sparse_csr_tensor(
        torch.arange(0, len(data_r) + 1, k, device=device),
        cols_r,
        data_r,
        size=S_dense.shape,
        dtype=dtype
    )
    
    # 2. Column Reordering (Centroid)
    perm_c = get_col_centroid_perm(S_sparse_r)
    
    # Apply col permutation: S @ P^T
    # This maps column j to perm_c[j]? No, perm_c is the list of columns in new order.
    # So old col j moves to new position `inv_perm_c[j]`.
    # Or, we want the matrix to have columns sorted.
    # New column k is old column perm_c[k].
    # So we read V in order perm_c. V_new = V[perm_c].
    # Matrix indices: if old col was C, new col is position of C in perm_c.
    
    # Invert permutation
    inv_perm_c = torch.zeros_like(perm_c)
    inv_perm_c[perm_c] = torch.arange(len(perm_c), device=device)
    
    # Update column indices in sparse matrix
    cols_rc = inv_perm_c[cols_r]
    
    # Rebuild sparse matrix with new column indices
    # NOTE: CSR requires sorted column indices per row for some ops? Torch might handle it or we sort.
    # To be safe, we should sort the cols per row.
    
    # We'll use COO to construct and convert to CSR to handle sorting
    S_sparse_rc_coo = torch.sparse_coo_tensor(
        torch.stack([rows_r, cols_rc]),
        data_r,
        size=S_dense.shape,
        dtype=dtype
    )
    S_sparse_rc = S_sparse_rc_coo.to_sparse_csr()
    
    # Reorder V
    V_reordered = V[perm_c]
    
    # Benchmark Reordered
    time_sparse_reordered = benchmark_sparse_matmul(S_sparse_rc, V_reordered, args.warmup, args.iters)
    print(f"Sparse Matmul (Reordered):     {time_sparse_reordered:.3f} ms")
    print(f"Sparse Speedup:                {time_sparse_base / time_sparse_reordered:.2f}x")
    
    # Verification
    # Y_base = S_sparse @ V
    # Y_reordered = S_sparse_rc @ V_reordered
    # We expect Y_reordered = P_r @ Y_base
    # So Y_base = P_r.T @ Y_reordered
    # P_r is a permutation of rows. Y_reordered[i] corresponds to row perm_r[i] of original.
    # So Y_reordered[new_row] = Y_base[perm_r[new_row]]
    
    print("Verifying sparse reordering correctness...")
    Y_base = torch.mm(S_sparse, V)
    Y_reordered = torch.mm(S_sparse_rc, V_reordered)
    
    # Un-permute rows of Y_reordered
    # Y_reordered_mapped_back = torch.zeros_like(Y_reordered)
    # Y_reordered_mapped_back[perm_r] = Y_reordered 
    # Wait: perm_r[i] is the original index of the i-th row in reordered matrix.
    # So Y_reordered[i] should be placed at perm_r[i] in reconstructed Y.
    Y_reconstructed = torch.zeros_like(Y_base)
    Y_reconstructed[perm_r] = Y_reordered
    
    diff_sparse = (Y_base - Y_reconstructed).abs().max()
    print(f"Max difference sparse vs reordered: {diff_sparse.item():.6e}")



if __name__ == "__main__":
    main()



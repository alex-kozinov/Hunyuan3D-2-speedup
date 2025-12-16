import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import torch
import time
import argparse
from pathlib import Path

# Try to import torch-sparse (optional, for cuSPARSE backend)
# try:
#     from torch_sparse import SparseTensor, matmul as sparse_matmul_pyg
#     TORCH_SPARSE_AVAILABLE = True
# except ImportError:
#     TORCH_SPARSE_AVAILABLE = False
#     SparseTensor = None
#     sparse_matmul_pyg = None

# Argument parsing
parser = argparse.ArgumentParser(description="Benchmark attention computation (dense vs sparse)")
parser.add_argument("--mode", type=str, choices=["dense", "sparse", "both"], default="both",
                    help="Computation mode: 'dense' (standard matmul), 'sparse' (CSR sparse), or 'both' for comparison")
parser.add_argument("--sparse-backend", type=str, 
                    choices=["csr", "coo", "bsr", "hybrid", "torch_sparse"], default="hybrid",
                    help="Sparse backend: 'csr' (Compressed Sparse Row), 'coo' (Coordinate), "
                         "'bsr' (Block Sparse Row), 'hybrid' (dense on thresholded), "
                         "'torch_sparse' (PyG torch-sparse with cuSPARSE) (default: hybrid)")
parser.add_argument("--block-size", type=int, default=16,
                    help="Block size for BSR format (default: 16)")
parser.add_argument("--eps", type=float, default=1e-3,
                    help="Threshold for sparsifying attention weights (default: 1e-3)")
parser.add_argument("--reorder", type=str, choices=["none", "cols", "rows", "both"], default="none",
                    help="Reorder sparse attention matrix axes to improve locality: "
                         "'cols' permutes keys/columns (also permutes V rows), "
                         "'rows' permutes queries/rows (and unpermutes output), "
                         "'both' does both. Default: none")
parser.add_argument("--reorder-method", type=str, 
                    choices=["centroid", "degree", "rcm", "hilbert", "cache_cluster", "profile", "minmax"], 
                    default="cache_cluster",
                    help="Heuristic for computing permutations from sparsity pattern. "
                         "'centroid' sorts by nonzero centroid, "
                         "'degree' sorts by nonzero count, "
                         "'rcm' uses Reverse Cuthill–McKee (bandwidth reduction), "
                         "'hilbert' uses Hilbert/Z-order (good for spatial data), "
                         "'cache_cluster' clusters columns accessed together (RECOMMENDED for cache), "
                         "'profile' minimizes avg row span, "
                         "'minmax' minimizes MAX row span (best for ultra-sparse). Default: cache_cluster")
parser.add_argument("--reorder-only", action="store_true",
                    help="If set and --reorder != none, run only the reordered sparse benchmark (skip baseline sparse).")
parser.add_argument("--reorder-per-head", action="store_true",
                    help="Compute separate reordering for each attention head (slower precompute, potentially better locality)")
parser.add_argument("--warmup", type=int, default=5,
                    help="Number of warmup iterations (default: 5)")
parser.add_argument("--iterations", type=int, default=25,
                    help="Number of benchmark iterations (default: 25)")
parser.add_argument("--device", type=str, default="auto",
                    help="Device to use: 'cpu', 'cuda', or 'auto' (default: auto)")
args = parser.parse_args()

# Directory containing the tensors
# tensor_dir = Path("hunyuan_accel")
tensor_dir = Path("outputs/attention_tensors")

# Load the attention matrices
print("Loading attention tensors...")
attention_Q_np = np.load(tensor_dir / "attention_Q.npy")
attention_K_np = np.load(tensor_dir / "attention_K.npy")
attention_V_np = np.load(tensor_dir / "attention_V.npy")

# Convert to PyTorch tensors
attention_Q = torch.from_numpy(attention_Q_np)
attention_K = torch.from_numpy(attention_K_np)
attention_V = torch.from_numpy(attention_V_np)

print(f"\nOriginal tensor shapes:")
print(f"  attention_Q: {attention_Q.shape}")
print(f"  attention_K: {attention_K.shape}")
print(f"  attention_V: {attention_V.shape}")

# Double the number of queries by concatenating Q with itself
print(f"\nDoubling queries by concatenating Q with itself...")
# Concatenate along the sequence dimension (second-to-last dimension)
attention_Q_doubled = torch.cat([attention_Q, attention_Q], dim=-2)
print(f"  Doubled Q shape: {attention_Q_doubled.shape}")

# Note: V stays the same! It corresponds to keys, not queries
print(f"  V shape (unchanged): {attention_V.shape}")

# Compute QK^T with doubled Q
print(f"\nComputing QK^T with doubled queries...")
# Q: (..., seq_q, d_k), K: (..., seq_k, d_k) -> QK^T: (..., seq_q, seq_k)
attention_QK_T = torch.matmul(attention_Q_doubled, attention_K.transpose(-2, -1))
print(f"  QK^T shape: {attention_QK_T.shape}")

# Apply softmax to get attention weights
print(f"Applying softmax...")
attention_weights = torch.nn.functional.softmax(attention_QK_T, dim=-1)
print(f"  attention_softmax_QK_T shape: {attention_weights.shape}")

print(f"\nFinal tensor shapes for benchmark:")
print(f"  attention_softmax_QK_T: {attention_weights.shape}")
print(f"  attention_V: {attention_V.shape}")
print(f"\nOriginal tensor dtypes:")
print(f"  attention_Q: {attention_Q.dtype}")
print(f"  attention_K: {attention_K.dtype}")
print(f"  attention_V: {attention_V.dtype}")

# Check if CUDA is available
if args.device == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args.device)
print(f"\nUsing device: {device}")

# Convert to float32 (sparse operations require float32, not half precision)
print(f"\nConverting to float32 (required for sparse operations)...")
print(f"  Original dtype: {attention_weights.dtype}")
attention_weights = attention_weights.to(dtype=torch.float32)
attention_V = attention_V.to(dtype=torch.float32)
print(f"  Converted to dtype: {attention_weights.dtype}")

# Move tensors to device
attention_weights = attention_weights.to(device)
attention_V = attention_V.to(device)

# Benchmark configuration
num_warmup = args.warmup
num_iterations = args.iterations
EPS = args.eps

def sparsity_stats(attn: torch.Tensor, eps: float):
    mask = attn.abs() > eps
    num_nonzero = int(mask.sum().item())
    total_elements = attn.numel()
    sparsity = 1.0 - (num_nonzero / total_elements)

    # Average NNZ per "row" along the last dimension (keys), averaged across all batches/heads/queries.
    # mask shape: (..., seq_q, seq_k) -> nnz_per_row: (..., seq_q)
    nnz_per_row = mask.sum(dim=-1).to(dtype=torch.float32)
    avg_nnz_per_row = float(nnz_per_row.mean().item())
    return num_nonzero, total_elements, sparsity, avg_nnz_per_row

# Analyze sparsity
num_nonzero, total_elements, sparsity, avg_nnz_per_row = sparsity_stats(attention_weights, EPS)

print(f"\n{'='*60}")
print(f"SPARSITY ANALYSIS (eps={EPS})")
print(f"{'='*60}")
print(f"Non-zero elements: {num_nonzero:,} / {total_elements:,}")
print(f"Sparsity: {sparsity*100:.4f}%")
print(f"Average non-zeros per row: {avg_nnz_per_row:.2f}")
print(f"{'='*60}")

def _aggregate_mask_2d(attn: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Aggregate a potentially batched attention tensor (..., seq_q, seq_k) into a single 2D
    boolean mask (seq_q, seq_k) by OR-ing across all leading dims.
    """
    if attn.dim() < 2:
        raise ValueError(f"Expected attention tensor with >=2 dims, got shape {tuple(attn.shape)}")
    mask = attn.abs() > eps
    if mask.dim() == 2:
        return mask
    seq_q, seq_k = mask.shape[-2], mask.shape[-1]
    mask_flat = mask.reshape(-1, seq_q, seq_k)
    return mask_flat.any(dim=0)

def _invert_permutation(perm: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype)
    return inv

def _compute_hilbert_reordering(mask2d: torch.Tensor):
    """
    Hilbert curve ordering for columns (approximated by Z-order/Morton code).
    Good if columns correspond to spatial locations (e.g., 3D voxel grid).
    """
    seq_q, seq_k = mask2d.shape
    
    # Try to factorize seq_k into 2D/3D grid
    # Common case: seq_k = H*W or H*W*D
    # For simplicity, use 2D: try to find factors close to sqrt(seq_k)
    import math
    side = int(math.sqrt(seq_k))
    while seq_k % side != 0 and side > 1:
        side -= 1
    
    if side == 1:
        # Can't factorize, fall back to degree ordering
        row_deg = mask2d.sum(dim=1)
        col_deg = mask2d.sum(dim=0)
        row_perm = torch.argsort(-row_deg, stable=True).to(torch.int64)
        col_perm = torch.argsort(-col_deg, stable=True).to(torch.int64)
        return row_perm, col_perm
    
    H = side
    W = seq_k // side
    
    # Compute Z-order (Morton code) for each column
    def morton_encode(y, x):
        """Interleave bits of x and y"""
        answer = 0
        for i in range(16):  # Assuming coordinates < 2^16
            answer |= ((x & (1 << i)) << i) | ((y & (1 << i)) << (i + 1))
        return answer
    
    morton_codes = []
    for col in range(seq_k):
        y = col // W
        x = col % W
        morton_codes.append(morton_encode(y, x))
    
    # Sort columns by Morton code
    col_perm = torch.tensor(sorted(range(seq_k), key=lambda i: morton_codes[i]), 
                            dtype=torch.int64, device=mask2d.device)
    
    # For rows, use degree ordering
    row_deg = mask2d.sum(dim=1)
    row_perm = torch.argsort(-row_deg, stable=True).to(torch.int64)
    
    return row_perm, col_perm

def _compute_cache_cluster_reordering(mask2d: torch.Tensor):
    """
    Cluster columns that are frequently accessed together.
    
    Strategy: Build a column-column similarity matrix based on how often
    columns appear together in the same query's nonzero set.
    Then use spectral clustering or greedy ordering.
    """
    seq_q, seq_k = mask2d.shape
    
    # Compute column-column co-occurrence matrix: how often cols i,j appear together
    # For large matrices, this is expensive; use approximation
    mask_float = mask2d.to(dtype=torch.float32)
    
    # C[i,j] = number of rows where both col i and col j are nonzero
    # C = mask^T @ mask (this is seq_k x seq_k, could be large!)
    if seq_k > 5000:
        # Too large, fall back to simpler method
        print(f"  (seq_k={seq_k} too large for full similarity matrix, using fast approximation)")
        return _compute_profile_minimization(mask2d)
    
    print(f"  Computing column similarity matrix ({seq_k}x{seq_k})...")
    C = torch.matmul(mask_float.T, mask_float)  # seq_k x seq_k
    
    # Normalize to similarity (Jaccard-like)
    col_deg = mask2d.sum(dim=0).to(torch.float32)  # seq_k
    # S[i,j] = C[i,j] / (deg[i] + deg[j] - C[i,j])
    # To avoid division, use simpler: S[i,j] = C[i,j] / sqrt(deg[i] * deg[j])
    deg_sqrt = torch.sqrt(col_deg + 1e-8)
    S = C / (deg_sqrt[:, None] * deg_sqrt[None, :] + 1e-8)
    
    # Greedy ordering: Start with highest-degree column, add most similar unvisited
    visited = torch.zeros(seq_k, dtype=torch.bool, device=mask2d.device)
    col_perm_list = []
    
    # Start with highest-degree column
    start_col = torch.argmax(col_deg).item()
    col_perm_list.append(start_col)
    visited[start_col] = True
    
    for _ in range(seq_k - 1):
        # Find unvisited column most similar to any visited column
        current = col_perm_list[-1]
        S_current = S[current].clone()
        S_current[visited] = -1  # Mask visited
        
        if S_current.max() <= 0:
            # No more similar columns, pick highest-degree unvisited
            remaining_deg = col_deg.clone()
            remaining_deg[visited] = -1
            next_col = torch.argmax(remaining_deg).item()
        else:
            next_col = torch.argmax(S_current).item()
        
        col_perm_list.append(next_col)
        visited[next_col] = True
    
    col_perm = torch.tensor(col_perm_list, dtype=torch.int64, device=mask2d.device)
    
    # For rows, use degree ordering
    row_deg = mask2d.sum(dim=1)
    row_perm = torch.argsort(-row_deg, stable=True).to(torch.int64)
    
    return row_perm, col_perm

def _compute_profile_minimization(mask2d: torch.Tensor):
    """
    Minimize profile: sum of (max_col_idx - min_col_idx) per row.
    
    Greedy heuristic: Order columns to minimize average span of rows accessing them.
    """
    seq_q, seq_k = mask2d.shape
    
    # For each column, compute the "spread" of rows that access it
    col_spread = []
    for j in range(seq_k):
        rows_accessing = mask2d[:, j].nonzero(as_tuple=False).squeeze(-1)
        if len(rows_accessing) == 0:
            col_spread.append((float('inf'), j))  # Isolated column, put at end
        else:
            spread = rows_accessing.max().item() - rows_accessing.min().item()
            col_spread.append((spread, j))
    
    # Sort columns by spread (ascending)
    col_spread_sorted = sorted(col_spread)
    col_perm = torch.tensor([j for _, j in col_spread_sorted], 
                            dtype=torch.int64, device=mask2d.device)
    
    # For rows, use degree ordering
    row_deg = mask2d.sum(dim=1)
    row_perm = torch.argsort(-row_deg, stable=True).to(torch.int64)
    
    return row_perm, col_perm

def _compute_minmax_reordering(mask2d: torch.Tensor):
    """
    Minimize MAXIMUM row span (best for ultra-sparse: avg nnz/row < 5).
    
    Fast approximation: Group columns by the rows they appear in,
    ordering groups to minimize max span.
    """
    seq_q, seq_k = mask2d.shape
    device = mask2d.device
    
    # For very large matrices, fall back to simpler method
    if seq_k > 5000 or seq_q > 10000:
        print("  (Matrix too large for exact minmax, using fast approximation)")
        return _compute_profile_minimization(mask2d)
    
    # Fast approach: For each column, compute its "impact" on max span
    # Impact = max distance from existing columns in rows where it appears
    
    col_deg = mask2d.sum(dim=0)  # How many rows each column appears in
    
    if col_deg.max().item() == 0:
        # All isolated
        return torch.arange(seq_q, dtype=torch.int64, device=device), \
               torch.arange(seq_k, dtype=torch.int64, device=device)
    
    # Convert to COO for faster access
    rows, cols = mask2d.nonzero(as_tuple=True)
    
    # Build a mapping: row -> list of columns
    from collections import defaultdict
    row_to_cols = defaultdict(list)
    for r, c in zip(rows.cpu().numpy(), cols.cpu().numpy()):
        row_to_cols[int(r)].append(int(c))
    
    # Start with highest-degree column
    start_col = torch.argmax(col_deg).item()
    placed = {start_col}
    col_perm_list = [start_col]
    
    # Greedy: add column that appears in most rows WITH already-placed columns
    # (This groups columns that co-occur, minimizing span)
    for _ in range(seq_k - 1):
        best_col = None
        best_score = -1
        
        # Only evaluate top candidates by degree (much faster)
        remaining_deg = col_deg.clone()
        for c in placed:
            remaining_deg[c] = -1
        
        # Top 100 candidates
        top_k = min(100, (remaining_deg >= 0).sum().item())
        if top_k == 0:
            break
        
        _, top_candidates = torch.topk(remaining_deg, top_k)
        
        for cand in top_candidates.cpu().numpy():
            cand = int(cand)
            if cand in placed:
                continue
            
            # Score: number of rows where this column co-occurs with placed columns
            cooccur_count = 0
            for row in range(seq_q):
                if row not in row_to_cols:
                    continue
                row_cols = row_to_cols[row]
                if cand in row_cols and any(c in placed for c in row_cols):
                    cooccur_count += 1
            
            # Tie-break by degree
            score = cooccur_count * 1000 + col_deg[cand].item()
            
            if score > best_score:
                best_score = score
                best_col = cand
        
        if best_col is None:
            # No more connected columns, add by degree
            for c in range(seq_k):
                if c not in placed:
                    best_col = c
                    break
        
        if best_col is not None:
            placed.add(best_col)
            col_perm_list.append(best_col)
    
    # Add any remaining columns (isolated)
    for c in range(seq_k):
        if c not in placed:
            col_perm_list.append(c)
    
    col_perm = torch.tensor(col_perm_list, dtype=torch.int64, device=device)
    
    # For rows, use degree ordering
    row_deg = mask2d.sum(dim=1)
    row_perm = torch.argsort(-row_deg, stable=True).to(torch.int64)
    
    return row_perm, col_perm

def compute_reordering_from_mask(mask2d: torch.Tensor, method: str = "centroid"):
    """
    Compute row/col permutations from a 2D boolean sparsity mask.
    Returns (row_perm, col_perm), each a 1D int64 tensor on the same device as mask2d.
    
    Key insight: For sparse_attn @ dense_V, the bottleneck is irregular V row accesses.
    Column reordering groups V rows that are accessed together → better cache locality.
    """
    if mask2d.dim() != 2:
        raise ValueError(f"mask2d must be 2D, got shape {tuple(mask2d.shape)}")
    seq_q, seq_k = mask2d.shape

    if method == "degree":
        row_deg = mask2d.sum(dim=1)
        col_deg = mask2d.sum(dim=0)
        # Descending degree, stable-ish tie-break by index.
        row_perm = torch.argsort(-row_deg, stable=True)
        col_perm = torch.argsort(-col_deg, stable=True)
        return row_perm.to(torch.int64), col_perm.to(torch.int64)
    
    if method == "hilbert":
        # Hilbert curve ordering: good if columns have spatial structure (e.g., 3D grid)
        # Map column index to 2D/3D coordinates, then Hilbert order
        # For now, approximate with Z-order (Morton code) which is simpler
        return _compute_hilbert_reordering(mask2d)
    
    if method == "cache_cluster":
        # Cluster columns that are frequently accessed together by the same queries
        return _compute_cache_cluster_reordering(mask2d)
    
    if method == "profile":
        # Minimize profile: sum of row span (max_col - min_col)
        return _compute_profile_minimization(mask2d)
    
    if method == "minmax":
        # Minimize MAXIMUM row span (best for ultra-sparse: avg nnz/row < 5)
        return _compute_minmax_reordering(mask2d)

    if method == "rcm":
        # Reverse Cuthill–McKee ordering on the bipartite graph:
        # Nodes: rows [0..seq_q-1], cols [seq_q..seq_q+seq_k-1]
        # Edge between row i and col (seq_q+j) iff mask2d[i,j] is True.
        #
        # We compute a single RCM ordering over all nodes, then extract row/col
        # permutations by preserving relative order among their node subsets.
        #
        # Note: runs on CPU (precompute only; excluded from timing).
        mask_cpu = mask2d.to(device="cpu", dtype=torch.bool)
        nz = mask_cpu.nonzero(as_tuple=False)
        if nz.numel() == 0:
            row_perm = torch.arange(seq_q, device=mask2d.device, dtype=torch.int64)
            col_perm = torch.arange(seq_k, device=mask2d.device, dtype=torch.int64)
            return row_perm, col_perm

        rows = nz[:, 0].to(dtype=torch.int64).numpy()
        cols = nz[:, 1].to(dtype=torch.int64).numpy()

        N = seq_q + seq_k
        # Degrees
        deg = np.zeros((N,), dtype=np.int64)
        np.add.at(deg, rows, 1)
        np.add.at(deg, seq_q + cols, 1)

        # Adjacency lists (python lists of numpy arrays to keep build simple)
        adj = [[] for _ in range(N)]
        for r, c in zip(rows, cols):
            cn = int(seq_q + c)
            rn = int(r)
            adj[rn].append(cn)
            adj[cn].append(rn)

        visited = np.zeros((N,), dtype=np.bool_)
        order = []

        # Helper: BFS from a start node, visiting neighbors in increasing degree.
        from collections import deque

        def bfs(start: int):
            q = deque([start])
            visited[start] = True
            while q:
                u = q.popleft()
                order.append(u)
                neigh = adj[u]
                if not neigh:
                    continue
                # Sort neighbors by degree (RCM heuristic)
                neigh_sorted = sorted(neigh, key=lambda x: (deg[x], x))
                for v in neigh_sorted:
                    if not visited[v]:
                        visited[v] = True
                        q.append(v)

        # Classic RCM: process connected components, starting from a minimum-degree node.
        # If graph disconnected (likely), repeat.
        remaining = np.where(~visited)[0]
        while remaining.size > 0:
            # Pick unvisited node with minimum degree (tie-break by index)
            start = int(remaining[np.argmin(deg[remaining])])
            bfs(start)
            remaining = np.where(~visited)[0]

        rcm = list(reversed(order))
        row_list = [n for n in rcm if n < seq_q]
        col_list = [n - seq_q for n in rcm if n >= seq_q]

        # If any isolated nodes ended up missing (shouldn't), append them.
        if len(row_list) != seq_q:
            missing = set(range(seq_q)) - set(row_list)
            row_list.extend(sorted(missing))
        if len(col_list) != seq_k:
            missing = set(range(seq_k)) - set(col_list)
            col_list.extend(sorted(missing))

        row_perm = torch.tensor(row_list, device=mask2d.device, dtype=torch.int64)
        col_perm = torch.tensor(col_list, device=mask2d.device, dtype=torch.int64)
        return row_perm, col_perm

    if method == "centroid":
        # Row centroid in column index space; column centroid in row index space.
        col_idx = torch.arange(seq_k, device=mask2d.device, dtype=torch.float32)
        row_idx = torch.arange(seq_q, device=mask2d.device, dtype=torch.float32)

        row_nnz = mask2d.sum(dim=1).to(torch.float32)
        col_nnz = mask2d.sum(dim=0).to(torch.float32)

        # Avoid div-by-zero; empty rows/cols get +inf centroid and end up at the end.
        row_nnz_safe = torch.clamp(row_nnz, min=1.0)
        col_nnz_safe = torch.clamp(col_nnz, min=1.0)

        row_centroid = (mask2d.to(torch.float32) * col_idx[None, :]).sum(dim=1) / row_nnz_safe
        col_centroid = (mask2d.to(torch.float32) * row_idx[:, None]).sum(dim=0) / col_nnz_safe

        row_centroid = torch.where(row_nnz > 0, row_centroid, torch.full_like(row_centroid, float("inf")))
        col_centroid = torch.where(col_nnz > 0, col_centroid, torch.full_like(col_centroid, float("inf")))

        row_perm = torch.argsort(row_centroid, stable=True)
        col_perm = torch.argsort(col_centroid, stable=True)
        return row_perm.to(torch.int64), col_perm.to(torch.int64)

    raise ValueError(f"Unknown reorder method: {method}")

def apply_reordering(attn: torch.Tensor, V: torch.Tensor, row_perm: torch.Tensor, col_perm: torch.Tensor, what: str):
    """
    Apply row/col reordering to attention (..., seq_q, seq_k) and V (..., seq_k, head_dim).
    Returns (attn_reordered, V_reordered, inv_row_perm_or_None).
    """
    attn_r = attn
    V_r = V
    inv_row_perm = None

    if what in ("rows", "both"):
        attn_r = attn_r.index_select(dim=-2, index=row_perm)
        inv_row_perm = _invert_permutation(row_perm)

    if what in ("cols", "both"):
        attn_r = attn_r.index_select(dim=-1, index=col_perm)
        V_r = V_r.index_select(dim=-2, index=col_perm)

    return attn_r, V_r, inv_row_perm

def _avg_row_span(mask2d: torch.Tensor) -> float:
    """
    Average (max_col - min_col) over rows with at least 1 nonzero.
    """
    stats = _row_span_stats(mask2d)
    return stats['avg']

def _row_span_stats(mask2d: torch.Tensor) -> dict:
    """
    Compute detailed row span statistics (avg, min, max, median, p95).
    Span = (max_col_idx - min_col_idx) for each row with at least 1 nonzero.
    """
    if mask2d.numel() == 0:
        return {'avg': 0.0, 'min': 0, 'max': 0, 'median': 0.0, 'p95': 0.0}
    
    seq_q, seq_k = mask2d.shape
    col_idx = torch.arange(seq_k, device=mask2d.device)
    
    # min index: use large fill for zeros then reduce
    big = torch.full((seq_q, seq_k), seq_k, device=mask2d.device, dtype=col_idx.dtype)
    min_idx = torch.min(torch.where(mask2d, col_idx[None, :], big), dim=1).values
    max_idx = torch.max(torch.where(mask2d, col_idx[None, :], torch.zeros_like(big)), dim=1).values
    has = mask2d.any(dim=1)
    
    if not has.any():
        return {'avg': 0.0, 'min': 0, 'max': 0, 'median': 0.0, 'p95': 0.0}
    
    span = (max_idx[has] - min_idx[has]).to(torch.float32)
    
    return {
        'avg': float(span.mean().item()),
        'min': int(span.min().item()),
        'max': int(span.max().item()),
        'median': float(torch.median(span).item()),
        'p95': float(torch.quantile(span, 0.95).item())
    }


def benchmark_dense(attention_weights, attention_V, num_warmup, num_iterations, device):
    """Benchmark dense matrix multiplication."""
    print(f"\n{'='*60}")
    print(f"BENCHMARKING DENSE ATTENTION COMPUTATION")
    print(f"{'='*60}")
    print(f"Warmup iterations: {num_warmup}")
    print(f"Benchmark iterations: {num_iterations}")
    
    # Warmup runs
    print("\nWarming up...")
    for _ in range(num_warmup):
        _ = torch.matmul(attention_weights, attention_V)
        if device.type == "cuda":
            torch.cuda.synchronize()
    
    # Benchmark runs
    print("Running benchmark...")
    times = []
    for i in range(num_iterations):
        start_time = time.perf_counter()
        attention_output = torch.matmul(attention_weights, attention_V)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return np.array(times), attention_output


def sparse_matmul_csr(attention_weights, attention_V, eps):
    """CSR (Compressed Sparse Row) format - good for row-wise operations."""
    attention_sparse = attention_weights.clone()
    attention_sparse[attention_sparse.abs() <= eps] = 0
    attention_sparse_csr = attention_sparse.to_sparse_csr()
    return torch.sparse.mm(attention_sparse_csr, attention_V)


def sparse_matmul_coo(attention_weights, attention_V, eps):
    """COO (Coordinate) format - flexible for irregular sparsity."""
    attention_sparse = attention_weights.clone()
    attention_sparse[attention_sparse.abs() <= eps] = 0
    attention_sparse_coo = attention_sparse.to_sparse()  # Default is COO
    return torch.sparse.mm(attention_sparse_coo, attention_V)


def sparse_matmul_bsr(attention_weights, attention_V, eps, block_size):
    """
    BSR (Block Sparse Row) format - efficient for block-structured sparsity.
    Good if sparsity follows block patterns (e.g., local neighborhoods in 3D grid).
    """
    seq_q, seq_k = attention_weights.shape[-2:]
    
    # Check if dimensions are compatible with block size
    if seq_q % block_size != 0 or seq_k % block_size != 0:
        # Fall back to CSR if not block-aligned
        return sparse_matmul_csr(attention_weights, attention_V, eps)
    
    attention_sparse = attention_weights.clone()
    attention_sparse[attention_sparse.abs() <= eps] = 0
    
    try:
        attention_sparse_bsr = attention_sparse.to_sparse_bsr(blocksize=(block_size, block_size))
        # BSR doesn't support torch.sparse.mm directly, convert to CSR
        attention_sparse_csr = attention_sparse_bsr.to_sparse_csr()
        return torch.sparse.mm(attention_sparse_csr, attention_V)
    except:
        # Fall back to CSR if BSR conversion fails
        return sparse_matmul_csr(attention_weights, attention_V, eps)


def sparse_matmul_hybrid(attention_weights, attention_V, eps):
    """
    Hybrid approach: Dense matmul on thresholded matrix.
    Often faster than true sparse for moderate sparsity or small matrices.
    """
    attention_thresholded = attention_weights.clone()
    attention_thresholded[attention_thresholded.abs() <= eps] = 0
    return torch.matmul(attention_thresholded, attention_V)


def sparse_matmul_torch_sparse(attention_weights, attention_V, eps):
    """
    torch-sparse backend: Uses PyTorch Geometric's SparseTensor (wraps cuSPARSE).
    Much more optimized than PyTorch native sparse, with better batched support.
    
    Requires: pip install torch-sparse
    """
    if not TORCH_SPARSE_AVAILABLE:
        raise ImportError(
            "torch-sparse is not installed. Install it with:\n"
            "pip install torch-sparse -f https://data.pyg.org/whl/torch-$(python -c 'import torch; print(torch.__version__)')+$(python -c 'import torch; print(torch.version.cuda)').html"
        )
    
    # Threshold to sparsify
    attention_sparse = attention_weights.clone()
    attention_sparse[attention_sparse.abs() <= eps] = 0
    
    # Convert to COO format (needed for SparseTensor)
    attention_coo = attention_sparse.to_sparse()
    indices = attention_coo.indices()  # [2, nnz]
    values = attention_coo.values()    # [nnz]
    
    # Create SparseTensor (torch-sparse format)
    # SparseTensor expects (row, col, value) not (indices, values)
    row = indices[0]
    col = indices[1]
    
    sparse_tensor = SparseTensor(
        row=row,
        col=col,
        value=values,
        sparse_sizes=(attention_weights.shape[0], attention_weights.shape[1])
    )
    
    # Perform sparse @ dense matmul using torch-sparse (uses cuSPARSE)
    # This is highly optimized compared to torch.sparse.mm
    output = sparse_matmul_pyg(sparse_tensor, attention_V)
    
    return output


def sparse_attention_matmul(attention_weights, attention_V, eps, device, backend="hybrid", block_size=16):
    """
    Perform sparse attention matmul with selectable backend.
    
    Backends:
    - 'csr': Compressed Sparse Row (good for row-wise operations)
    - 'coo': Coordinate format (flexible for irregular patterns)
    - 'bsr': Block Sparse Row (efficient for block-structured sparsity)
    - 'hybrid': Dense matmul on thresholded matrix (often fastest for moderate sparsity)
    - 'torch_sparse': PyTorch Geometric SparseTensor (uses cuSPARSE, optimized)
    """
    original_attn_shape = attention_weights.shape
    original_V_shape = attention_V.shape
    
    # Select sparse matmul function
    sparse_funcs = {
        'csr': lambda a, v: sparse_matmul_csr(a, v, eps),
        'coo': lambda a, v: sparse_matmul_coo(a, v, eps),
        'bsr': lambda a, v: sparse_matmul_bsr(a, v, eps, block_size),
        'hybrid': lambda a, v: sparse_matmul_hybrid(a, v, eps),
        'torch_sparse': lambda a, v: sparse_matmul_torch_sparse(a, v, eps),
    }
    
    if backend not in sparse_funcs:
        raise ValueError(f"Unknown backend: {backend}")
    
    sparse_func = sparse_funcs[backend]
    
    # Get dimensions
    if attention_weights.dim() == 2:
        # Simple 2D case
        return sparse_func(attention_weights, attention_V)
    
    else:
        # Multi-dimensional case: Use VECTORIZED operations
        *leading_dims_attn, seq_q, seq_k = original_attn_shape
        *leading_dims_V, seq_k_v, head_dim = original_V_shape
        
        # Sanity check
        if seq_k != seq_k_v:
            raise ValueError(f"Attention seq_k ({seq_k}) must match V seq_k ({seq_k_v})")
        
        # Flatten all leading dimensions
        num_batches_attn = int(np.prod(leading_dims_attn)) if leading_dims_attn else 1
        num_batches_V = int(np.prod(leading_dims_V)) if leading_dims_V else 1
        
        if num_batches_attn != num_batches_V:
            raise ValueError(f"Batch dimensions don't match: attention {leading_dims_attn} vs V {leading_dims_V}")
        
        # For batched tensors, hybrid is usually fastest (vectorized operations)
        if backend == "hybrid":
            # OPTIMIZATION: Vectorized thresholding + torch.bmm
            attn_thresholded = attention_weights.clone()
            attn_thresholded[attn_thresholded.abs() <= eps] = 0
            
            # Reshape for batched matmul
            attn_flat = attn_thresholded.reshape(num_batches_attn, seq_q, seq_k)
            V_flat = attention_V.reshape(num_batches_V, seq_k, head_dim)
            
            # Use torch.bmm (highly optimized)
            output_flat = torch.bmm(attn_flat, V_flat)
            
            # Reshape back
            output_shape = leading_dims_attn + [seq_q, head_dim]
            return output_flat.reshape(output_shape)
        else:
            # For true sparse formats, process each batch (no batched sparse matmul in PyTorch)
            attn_flat = attention_weights.reshape(num_batches_attn, seq_q, seq_k)
            V_flat = attention_V.reshape(num_batches_V, seq_k, head_dim)
            
            outputs = []
            for b in range(num_batches_attn):
                out_b = sparse_func(attn_flat[b], V_flat[b])
                outputs.append(out_b)
            
            output_flat = torch.stack(outputs, dim=0)
            output_shape = leading_dims_attn + [seq_q, head_dim]
            return output_flat.reshape(output_shape)


def benchmark_sparse(attention_weights, attention_V, num_warmup, num_iterations, device, eps, backend="hybrid", block_size=16):
    """
    Benchmark sparse/thresholded attention computation with selectable backend.
    
    Backends:
    - 'csr': Compressed Sparse Row (row-wise efficient)
    - 'coo': Coordinate format (flexible)
    - 'bsr': Block Sparse Row (block-structured sparsity)
    - 'hybrid': Dense matmul on thresholded matrix (often fastest)
    """
    print(f"\n{'='*60}")
    print(f"BENCHMARKING SPARSE/THRESHOLDED ATTENTION COMPUTATION")
    print(f"{'='*60}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"V shape: {attention_V.shape}")
    print(f"Tensor dimensions: {attention_weights.dim()}D")
    print(f"Sparse backend: {backend.upper()}")
    if backend == "bsr":
        print(f"Block size: {block_size}x{block_size}")
    
    # Describe backend
    backend_desc = {
        'csr': "CSR (Compressed Sparse Row) - efficient for row-wise access",
        'coo': "COO (Coordinate) - flexible for irregular sparsity",
        'bsr': f"BSR (Block Sparse Row {block_size}x{block_size}) - efficient for block patterns",
        'hybrid': "Hybrid (dense matmul on thresholded matrix) - vectorized operations",
        'torch_sparse': "torch-sparse (PyG SparseTensor with cuSPARSE) - optimized sparse ops"
    }
    print(f"Strategy: {backend_desc.get(backend, 'Unknown')}")
    
    print(f"Warmup iterations: {num_warmup}")
    print(f"Benchmark iterations: {num_iterations}")
    print(f"Sparsity threshold (eps): {eps}")
    
    # Warmup runs
    print("\nWarming up...")
    for _ in range(num_warmup):
        _ = sparse_attention_matmul(attention_weights, attention_V, eps, device, backend, block_size)
        if device.type == "cuda":
            torch.cuda.synchronize()
    
    # Benchmark runs
    print("Running benchmark...")
    times = []
    
    for i in range(num_iterations):
        start_time = time.perf_counter()
        attention_output = sparse_attention_matmul(attention_weights, attention_V, eps, device, backend, block_size)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return np.array(times), attention_output


def _equivalent_flops(attention_weights: torch.Tensor, attention_V: torch.Tensor) -> float:
    # attention: (..., seq_q, seq_k), V: (..., seq_k, head_dim)
    if attention_weights.shape[-1] != attention_V.shape[-2]:
        raise ValueError(f"seq_k mismatch: attn {attention_weights.shape} vs V {attention_V.shape}")
    seq_q = attention_weights.shape[-2]
    seq_k = attention_weights.shape[-1]
    head_dim = attention_V.shape[-1]
    leading = int(np.prod(attention_weights.shape[:-2])) if attention_weights.dim() > 2 else 1
    # matmul flops per batch: 2*seq_q*seq_k*head_dim
    return float(2 * leading * seq_q * seq_k * head_dim)

def print_benchmark_results(times, label, attention_weights, attention_V):
    """Print benchmark statistics with robust handling of variance."""
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    median_time = np.median(times)
    
    # Percentiles for robustness
    p25 = np.percentile(times, 25)
    p75 = np.percentile(times, 75)
    p95 = np.percentile(times, 95)
    
    # Trimmed mean (drop top/bottom 10%)
    if len(times) >= 10:
        sorted_times = np.sort(times)
        trim_count = max(1, len(times) // 10)
        trimmed_times = sorted_times[trim_count:-trim_count]
        trimmed_mean = np.mean(trimmed_times)
    else:
        trimmed_mean = mean_time
    
    # Coefficient of variation
    cv = (std_time / mean_time) * 100 if mean_time > 0 else 0
    
    print(f"\n{label} RESULTS:")
    print(f"  Mean time:       {mean_time*1000:.4f} ms (±{std_time*1000:.4f} ms, CV={cv:.1f}%)")
    print(f"  Trimmed mean:    {trimmed_mean*1000:.4f} ms (drop top/bottom 10%)")
    print(f"  Median time:     {median_time*1000:.4f} ms")
    print(f"  Min / Max:       {min_time*1000:.4f} / {max_time*1000:.4f} ms")
    print(f"  Percentiles:     P25={p25*1000:.4f} ms, P75={p75*1000:.4f} ms, P95={p95*1000:.4f} ms")
    
    # Compute throughput (dense-equivalent flops) using trimmed mean for robustness
    flops = _equivalent_flops(attention_weights, attention_V)
    throughput_mean = flops / mean_time / 1e9
    throughput_trimmed = flops / trimmed_mean / 1e9
    print(f"  Throughput:      {throughput_mean:.2f} GFLOPS (mean), {throughput_trimmed:.2f} GFLOPS (trimmed)")
    
    return trimmed_mean  # Return trimmed mean for more robust comparison


# Run benchmarks based on mode
results = {}

if args.mode in ["dense", "both"]:
    dense_times, dense_output = benchmark_dense(
        attention_weights, attention_V, num_warmup, num_iterations, device
    )
    results["dense"] = {
        "times": dense_times,
        "output": dense_output,
        "mean": print_benchmark_results(dense_times, "DENSE", attention_weights, attention_V)
    }

if args.mode in ["sparse", "both"]:
    # Baseline sparse
    if not (args.reorder != "none" and args.reorder_only):
        sparse_times, sparse_output = benchmark_sparse(
            attention_weights, attention_V, num_warmup, num_iterations, device, EPS,
            backend=args.sparse_backend, block_size=args.block_size
        )
        results["sparse"] = {
            "times": sparse_times,
            "output": sparse_output,
            "mean": print_benchmark_results(sparse_times, f"SPARSE ({args.sparse_backend.upper()})",
                                            attention_weights, attention_V)
        }

    # Reordered sparse
    if args.reorder != "none":
        print(f"\n{'='*60}")
        print("COMPUTING REORDERING (precompute; excluded from timing)")
        print(f"{'='*60}")
        mask2d = _aggregate_mask_2d(attention_weights, EPS)
        print(f"Reorder: {args.reorder} | Method: {args.reorder_method} | Aggregated mask shape: {tuple(mask2d.shape)}")
        
        # Detailed span analysis
        span_stats_before = _row_span_stats(mask2d)
        row_perm, col_perm = compute_reordering_from_mask(mask2d, method=args.reorder_method)
        mask2d_re = mask2d.index_select(0, row_perm).index_select(1, col_perm)
        span_stats_after = _row_span_stats(mask2d_re)
        
        print(f"\nRow span statistics (max_col_idx - min_col_idx per row):")
        print(f"  BEFORE reordering:")
        print(f"    Avg: {span_stats_before['avg']:.2f} | Median: {span_stats_before['median']:.2f} | " +
              f"Min: {span_stats_before['min']} | Max: {span_stats_before['max']} | P95: {span_stats_before['p95']:.2f}")
        print(f"  AFTER reordering:")
        print(f"    Avg: {span_stats_after['avg']:.2f} | Median: {span_stats_after['median']:.2f} | " +
              f"Min: {span_stats_after['min']} | Max: {span_stats_after['max']} | P95: {span_stats_after['p95']:.2f}")
        print(f"  IMPROVEMENT:")
        avg_reduction = (span_stats_before['avg'] - span_stats_after['avg']) / span_stats_before['avg'] * 100 if span_stats_before['avg'] > 0 else 0
        max_reduction = (span_stats_before['max'] - span_stats_after['max']) / span_stats_before['max'] * 100 if span_stats_before['max'] > 0 else 0
        print(f"    Avg span reduced by {avg_reduction:.1f}% | Max span reduced by {max_reduction:.1f}%")

        attn_r, V_r, inv_row_perm = apply_reordering(attention_weights, attention_V, row_perm, col_perm, args.reorder)

        def benchmark_sparse_reordered(attn_r, V_r, inv_row_perm):
            print(f"\n{'='*60}")
            print(f"BENCHMARKING REORDERED SPARSE/THRESHOLDED ATTENTION COMPUTATION")
            print(f"{'='*60}")
            print(f"Reorder: {args.reorder} | Method: {args.reorder_method} | Backend: {args.sparse_backend.upper()}")
            print(f"Warmup iterations: {num_warmup}")
            print(f"Benchmark iterations: {num_iterations}")
            print(f"Sparsity threshold (eps): {EPS}")

            print("\nWarming up...")
            for _ in range(num_warmup):
                out_r = sparse_attention_matmul(attn_r, V_r, EPS, device, args.sparse_backend, args.block_size)
                if inv_row_perm is not None:
                    out_r = out_r.index_select(dim=-2, index=inv_row_perm)
                if device.type == "cuda":
                    torch.cuda.synchronize()

            print("Running benchmark...")
            times = []
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                out_r = sparse_attention_matmul(attn_r, V_r, EPS, device, args.sparse_backend, args.block_size)
                if inv_row_perm is not None:
                    out_r = out_r.index_select(dim=-2, index=inv_row_perm)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            return np.array(times), out_r

        sparse_r_times, sparse_r_output = benchmark_sparse_reordered(attn_r, V_r, inv_row_perm)
        results["sparse_reordered"] = {
            "times": sparse_r_times,
            "output": sparse_r_output,
            "mean": print_benchmark_results(sparse_r_times, f"SPARSE REORDERED ({args.sparse_backend.upper()} | {args.reorder} | {args.reorder_method})",
                                            attention_weights, attention_V)
        }

# Comparison summary
if args.mode == "both":
    print(f"\n{'='*60}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*60}")
    dense_mean = results["dense"]["mean"]
    sparse_mean = results["sparse"]["mean"] if "sparse" in results else None
    
    print(f"Dense mean time:           {dense_mean*1000:.4f} ms")
    if sparse_mean is not None:
        print(f"Sparse ({args.sparse_backend.upper()}) mean time: {sparse_mean*1000:.4f} ms")
        speedup = dense_mean / sparse_mean
        if speedup >= 1:
            print(f"\nSpeedup (dense / sparse):  {speedup:.2f}x (sparse is faster)")
        else:
            print(f"\nSlowdown (sparse / dense): {1/speedup:.2f}x (dense is faster)")
    if "sparse_reordered" in results:
        sparse_r_mean = results["sparse_reordered"]["mean"]
        print(f"Sparse REORDERED mean time: {sparse_r_mean*1000:.4f} ms")
        speedup_r = dense_mean / sparse_r_mean
        if speedup_r >= 1:
            print(f"Speedup (dense / sparse_reordered):  {speedup_r:.2f}x (reordered sparse is faster)")
        else:
            print(f"Slowdown (sparse_reordered / dense): {1/speedup_r:.2f}x (dense is faster)")
        if sparse_mean is not None:
            rr = sparse_mean / sparse_r_mean
            if rr >= 1:
                print(f"\nReordering speedup (baseline_sparse / reordered_sparse): {rr:.2f}x")
            else:
                print(f"\nReordering slowdown (reordered_sparse / baseline_sparse): {1/rr:.2f}x")
    
    # Check output similarity
    dense_out = results["dense"]["output"]
    sparse_out = results["sparse"]["output"] if "sparse" in results else None
    
    def _print_diff_stats(label: str, out: torch.Tensor):
        abs_diff = torch.abs(dense_out - out)
        mae = torch.mean(abs_diff).item()
        max_diff = torch.max(abs_diff).item()
        mse = torch.mean((dense_out - out) ** 2).item()
        rmse = float(np.sqrt(mse))
        dense_mean_abs = torch.mean(torch.abs(dense_out)).item()
        relative_mae = mae / (dense_mean_abs + 1e-10)
        print(f"\nOutput difference vs DENSE ({label}):")
        print(f"  MAE (Mean Absolute Error):    {mae:.6e}")
        print(f"  Max absolute difference:      {max_diff:.6e}")
        print(f"  RMSE (Root Mean Squared Err): {rmse:.6e}")
        print(f"  Relative MAE:                 {relative_mae*100:.4f}%")
        print(f"  Dense output mean abs value:  {dense_mean_abs:.6e}")

    if sparse_out is not None:
        _print_diff_stats(f"sparse @ eps={EPS} ({args.sparse_backend.upper()})", sparse_out)
    if "sparse_reordered" in results:
        _print_diff_stats(f"sparse_reordered @ eps={EPS} ({args.sparse_backend.upper()} | {args.reorder} | {args.reorder_method})",
                          results["sparse_reordered"]["output"])
    print(f"{'='*60}")
    
    attention_output = dense_output
else:
    if args.mode == "sparse" and "sparse" not in results and "sparse_reordered" in results:
        attention_output = results["sparse_reordered"]["output"]
    else:
        attention_output = results[args.mode]["output"]

# Compute the final attention output
print("\nFinal attention output computed.")

print(f"\nAttention output shape: {attention_output.shape}")
print(f"Attention output dtype: {attention_output.dtype}")
print(f"Attention output device: {attention_output.device}")

# Print some statistics
print(f"\nAttention output statistics:")
print(f"  Min: {attention_output.min().item():.6f}")
print(f"  Max: {attention_output.max().item():.6f}")
print(f"  Mean: {attention_output.mean().item():.6f}")
print(f"  Std: {attention_output.std().item():.6f}")
print(f"  Non-zero elements: {torch.sum(attention_output.abs() > 1e-6).item()} / {attention_output.numel()}")
print(f"  Sparsity: {(1 - torch.sum(attention_output.abs() > 1e-6).item() / attention_output.numel()) * 100:.2f}%")

# Move back to CPU for saving
attention_output_cpu = attention_output.cpu()

# Save the output
output_path = tensor_dir / "attention_output.npy"
np.save(output_path, attention_output_cpu.numpy())
print(f"\nSaved attention output to: {output_path}")

# Also save as PyTorch tensor for convenience
output_pt_path = "attention_output.pt"
torch.save(attention_output_cpu, output_pt_path)
print(f"Saved attention output (PyTorch) to: {output_pt_path}")

# Print usage examples
print(f"\n{'='*60}")
print("USAGE EXAMPLES:")
print(f"{'='*60}")
print("Basic:")
print("  Dense only:          python compute_attention_output.py --mode dense")
print("  Sparse (hybrid):     python compute_attention_output.py --mode sparse --sparse-backend hybrid")
print("  Sparse (CSR):        python compute_attention_output.py --mode sparse --sparse-backend csr")
print("  Sparse (torch_sparse): python compute_attention_output.py --mode sparse --sparse-backend torch_sparse")
print("  Compare backends:    python compute_attention_output.py --mode both --sparse-backend csr")
print("")
print("Advanced reordering (improves cache locality for sparse @ dense_V):")
print("  Cache clustering:    python compute_attention_output.py --mode sparse --sparse-backend csr --reorder cols --reorder-method cache_cluster")
print("  MinMax (ultra-sparse): python compute_attention_output.py --mode sparse --sparse-backend csr --reorder cols --reorder-method minmax")
print("  RCM (bandwidth):     python compute_attention_output.py --mode sparse --sparse-backend csr --reorder both --reorder-method rcm")
print("  Hilbert (spatial):   python compute_attention_output.py --mode sparse --sparse-backend csr --reorder cols --reorder-method hilbert")
print("  Profile minimize:    python compute_attention_output.py --mode sparse --sparse-backend csr --reorder cols --reorder-method profile")
print("  Per-head reordering: python compute_attention_output.py --mode sparse --sparse-backend csr --reorder cols --reorder-method cache_cluster --reorder-per-head")
print("  Compare w/ baseline: python compute_attention_output.py --mode sparse --sparse-backend csr --reorder cols --reorder-method cache_cluster")
print("  Reorder only:        python compute_attention_output.py --mode sparse --sparse-backend csr --reorder cols --reorder-only --reorder-method cache_cluster")
print("")
print("Other:")
print("  Custom eps:          python compute_attention_output.py --mode sparse --eps 1e-4")
print("  More iterations:     python compute_attention_output.py --iterations 100 --warmup 10")
print(f"{'='*60}")


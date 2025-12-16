import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import torch
import time
import argparse
from pathlib import Path

# Argument parsing
parser = argparse.ArgumentParser(description="Compare all sparse attention backends")
parser.add_argument("--eps", type=float, default=1e-3,
                    help="Threshold for sparsifying attention weights (default: 1e-3)")
parser.add_argument("--block-size", type=int, default=16,
                    help="Block size for BSR format (default: 16)")
parser.add_argument("--warmup", type=int, default=5,
                    help="Number of warmup iterations (default: 5)")
parser.add_argument("--iterations", type=int, default=25,
                    help="Number of benchmark iterations (default: 25)")
parser.add_argument("--device", type=str, default="auto",
                    help="Device to use: 'cpu', 'cuda', or 'auto' (default: auto)")
parser.add_argument("--include-reordering", action="store_true",
                    help="Include reordering techniques in comparison (slower but more comprehensive)")
parser.add_argument("--reorder", type=str, choices=["none", "cols", "rows", "both"], default="cols",
                    help="Reorder axes: 'cols' (keys/V), 'rows' (queries), 'both', or 'none' (default: cols)")
args = parser.parse_args()

# Directory containing the tensors
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
attention_Q_doubled = torch.cat([attention_Q, attention_Q], dim=-2)
print(f"  Doubled Q shape: {attention_Q_doubled.shape}")

# Compute QK^T with doubled Q
print(f"\nComputing QK^T with doubled queries...")
attention_QK_T = torch.matmul(attention_Q_doubled, attention_K.transpose(-2, -1))
print(f"  QK^T shape: {attention_QK_T.shape}")

# Apply softmax to get attention weights
print(f"Applying softmax...")
attention_weights = torch.nn.functional.softmax(attention_QK_T, dim=-1)
print(f"  attention_softmax_QK_T shape: {attention_weights.shape}")

# Check if CUDA is available
if args.device == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args.device)
print(f"\nUsing device: {device}")

# Convert to float32 (sparse operations require float32)
print(f"\nConverting to float32...")
attention_weights = attention_weights.to(dtype=torch.float32)
attention_V = attention_V.to(dtype=torch.float32)

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

# ============================================================================
# Reordering functions (for cache locality optimization)
# ============================================================================

def _aggregate_mask_2d(attn: torch.Tensor, eps: float) -> torch.Tensor:
    """Aggregate a batched attention tensor into a single 2D boolean mask."""
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

def _compute_cache_cluster_reordering(mask2d: torch.Tensor):
    """Cluster columns that are frequently accessed together."""
    seq_q, seq_k = mask2d.shape
    
    mask_float = mask2d.to(dtype=torch.float32)
    
    if seq_k > 5000:
        print(f"  (seq_k={seq_k} too large for full similarity matrix, using degree ordering)")
        col_deg = mask2d.sum(dim=0)
        row_deg = mask2d.sum(dim=1)
        row_perm = torch.argsort(-row_deg, stable=True).to(torch.int64)
        col_perm = torch.argsort(-col_deg, stable=True).to(torch.int64)
        return row_perm, col_perm
    
    print(f"  Computing column similarity matrix ({seq_k}x{seq_k})...")
    C = torch.matmul(mask_float.T, mask_float)  # seq_k x seq_k
    
    col_deg = mask2d.sum(dim=0).to(torch.float32)  # seq_k
    deg_sqrt = torch.sqrt(col_deg + 1e-8)
    S = C / (deg_sqrt[:, None] * deg_sqrt[None, :] + 1e-8)
    
    # Greedy ordering
    visited = torch.zeros(seq_k, dtype=torch.bool, device=mask2d.device)
    col_perm_list = []
    
    start_col = torch.argmax(col_deg).item()
    col_perm_list.append(start_col)
    visited[start_col] = True
    
    for _ in range(seq_k - 1):
        current = col_perm_list[-1]
        S_current = S[current].clone()
        S_current[visited] = -1
        
        if S_current.max() <= 0:
            remaining_deg = col_deg.clone()
            remaining_deg[visited] = -1
            next_col = torch.argmax(remaining_deg).item()
        else:
            next_col = torch.argmax(S_current).item()
        
        col_perm_list.append(next_col)
        visited[next_col] = True
    
    col_perm = torch.tensor(col_perm_list, dtype=torch.int64, device=mask2d.device)
    
    row_deg = mask2d.sum(dim=1)
    row_perm = torch.argsort(-row_deg, stable=True).to(torch.int64)
    
    return row_perm, col_perm

def _compute_profile_minimization(mask2d: torch.Tensor):
    """Minimize profile: sum of (max_col_idx - min_col_idx) per row."""
    seq_q, seq_k = mask2d.shape
    
    col_spread = []
    for j in range(seq_k):
        rows_accessing = mask2d[:, j].nonzero(as_tuple=False).squeeze(-1)
        if len(rows_accessing) == 0:
            col_spread.append((float('inf'), j))
        else:
            spread = rows_accessing.max().item() - rows_accessing.min().item()
            col_spread.append((spread, j))
    
    col_spread_sorted = sorted(col_spread)
    col_perm = torch.tensor([j for _, j in col_spread_sorted], 
                            dtype=torch.int64, device=mask2d.device)
    
    row_deg = mask2d.sum(dim=1)
    row_perm = torch.argsort(-row_deg, stable=True).to(torch.int64)
    
    return row_perm, col_perm

def _compute_minmax_reordering(mask2d: torch.Tensor):
    """Minimize MAXIMUM row span (best for ultra-sparse)."""
    seq_q, seq_k = mask2d.shape
    device = mask2d.device
    
    if seq_k > 5000 or seq_q > 10000:
        print("  (Matrix too large for exact minmax, using fast approximation)")
        return _compute_profile_minimization(mask2d)
    
    col_deg = mask2d.sum(dim=0)
    
    if col_deg.max().item() == 0:
        return torch.arange(seq_q, dtype=torch.int64, device=device), \
               torch.arange(seq_k, dtype=torch.int64, device=device)
    
    rows, cols = mask2d.nonzero(as_tuple=True)
    
    from collections import defaultdict
    row_to_cols = defaultdict(list)
    for r, c in zip(rows.cpu().numpy(), cols.cpu().numpy()):
        row_to_cols[int(r)].append(int(c))
    
    start_col = torch.argmax(col_deg).item()
    placed = {start_col}
    col_perm_list = [start_col]
    
    for _ in range(seq_k - 1):
        best_col = None
        best_score = -1
        
        remaining_deg = col_deg.clone()
        for c in placed:
            remaining_deg[c] = -1
        
        top_k = min(100, (remaining_deg >= 0).sum().item())
        if top_k == 0:
            break
        
        _, top_candidates = torch.topk(remaining_deg, top_k)
        
        for cand in top_candidates.cpu().numpy():
            cand = int(cand)
            if cand in placed:
                continue
            
            cooccur_count = 0
            for row in range(seq_q):
                if row not in row_to_cols:
                    continue
                row_cols = row_to_cols[row]
                if cand in row_cols and any(c in placed for c in row_cols):
                    cooccur_count += 1
            
            score = cooccur_count * 1000 + col_deg[cand].item()
            
            if score > best_score:
                best_score = score
                best_col = cand
        
        if best_col is None:
            for c in range(seq_k):
                if c not in placed:
                    best_col = c
                    break
        
        if best_col is not None:
            placed.add(best_col)
            col_perm_list.append(best_col)
    
    for c in range(seq_k):
        if c not in placed:
            col_perm_list.append(c)
    
    col_perm = torch.tensor(col_perm_list, dtype=torch.int64, device=device)
    
    row_deg = mask2d.sum(dim=1)
    row_perm = torch.argsort(-row_deg, stable=True).to(torch.int64)
    
    return row_perm, col_perm

def compute_reordering_from_mask(mask2d: torch.Tensor, method: str = "cache_cluster"):
    """Compute row/col permutations from a 2D boolean sparsity mask."""
    if mask2d.dim() != 2:
        raise ValueError(f"mask2d must be 2D, got shape {tuple(mask2d.shape)}")
    seq_q, seq_k = mask2d.shape

    if method == "degree":
        row_deg = mask2d.sum(dim=1)
        col_deg = mask2d.sum(dim=0)
        row_perm = torch.argsort(-row_deg, stable=True)
        col_perm = torch.argsort(-col_deg, stable=True)
        return row_perm.to(torch.int64), col_perm.to(torch.int64)
    
    if method == "cache_cluster":
        return _compute_cache_cluster_reordering(mask2d)
    
    if method == "profile":
        return _compute_profile_minimization(mask2d)
    
    if method == "minmax":
        return _compute_minmax_reordering(mask2d)
    
    if method == "centroid":
        col_idx = torch.arange(seq_k, device=mask2d.device, dtype=torch.float32)
        row_idx = torch.arange(seq_q, device=mask2d.device, dtype=torch.float32)

        row_nnz = mask2d.sum(dim=1).to(torch.float32)
        col_nnz = mask2d.sum(dim=0).to(torch.float32)

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
    """Apply row/col reordering to attention and V."""
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

def _row_span_stats(mask2d: torch.Tensor) -> dict:
    """Compute detailed row span statistics."""
    if mask2d.numel() == 0:
        return {'avg': 0.0, 'min': 0, 'max': 0, 'median': 0.0, 'p95': 0.0}
    
    seq_q, seq_k = mask2d.shape
    col_idx = torch.arange(seq_k, device=mask2d.device)
    
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

# ============================================================================
# Sparse matmul implementations
# ============================================================================

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
    """BSR (Block Sparse Row) format - efficient for block-structured sparsity."""
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
    """Hybrid approach: Dense matmul on thresholded matrix."""
    attention_thresholded = attention_weights.clone()
    attention_thresholded[attention_thresholded.abs() <= eps] = 0
    return torch.matmul(attention_thresholded, attention_V)

def sparse_attention_matmul(attention_weights, attention_V, eps, device, backend="hybrid", block_size=16):
    """Perform sparse attention matmul with selectable backend."""
    original_attn_shape = attention_weights.shape
    original_V_shape = attention_V.shape
    
    # Select sparse matmul function
    sparse_funcs = {
        'csr': lambda a, v: sparse_matmul_csr(a, v, eps),
        'coo': lambda a, v: sparse_matmul_coo(a, v, eps),
        'bsr': lambda a, v: sparse_matmul_bsr(a, v, eps, block_size),
        'hybrid': lambda a, v: sparse_matmul_hybrid(a, v, eps),
    }
    
    if backend not in sparse_funcs:
        raise ValueError(f"Unknown backend: {backend}")
    
    sparse_func = sparse_funcs[backend]
    
    # Get dimensions
    if attention_weights.dim() == 2:
        # Simple 2D case
        return sparse_func(attention_weights, attention_V)
    else:
        # Multi-dimensional case
        *leading_dims_attn, seq_q, seq_k = original_attn_shape
        *leading_dims_V, seq_k_v, head_dim = original_V_shape
        
        # Sanity check
        if seq_k != seq_k_v:
            raise ValueError(f"Attention seq_k ({seq_k}) must match V seq_k ({seq_k_v})")
        
        # Flatten all leading dimensions
        num_batches_attn = int(np.prod(leading_dims_attn)) if leading_dims_attn else 1
        num_batches_V = int(np.prod(leading_dims_V)) if leading_dims_V else 1
        
        if num_batches_attn != num_batches_V:
            raise ValueError(f"Batch dimensions don't match")
        
        # For batched tensors, hybrid is usually fastest (vectorized operations)
        if backend == "hybrid":
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
            # For true sparse formats, process each batch
            attn_flat = attention_weights.reshape(num_batches_attn, seq_q, seq_k)
            V_flat = attention_V.reshape(num_batches_V, seq_k, head_dim)
            
            outputs = []
            for b in range(num_batches_attn):
                out_b = sparse_func(attn_flat[b], V_flat[b])
                outputs.append(out_b)
            
            output_flat = torch.stack(outputs, dim=0)
            output_shape = leading_dims_attn + [seq_q, head_dim]
            return output_flat.reshape(output_shape)

# ============================================================================
# Benchmark functions
# ============================================================================

def benchmark_dense(attention_weights, attention_V, num_warmup, num_iterations, device):
    """Benchmark dense matrix multiplication."""
    print(f"\n{'='*60}")
    print(f"BENCHMARKING DENSE ATTENTION (baseline)")
    print(f"{'='*60}")
    
    # Warmup runs
    print("Warming up...")
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

def benchmark_sparse(attention_weights, attention_V, num_warmup, num_iterations, device, eps, backend="hybrid", block_size=16):
    """Benchmark sparse/thresholded attention computation."""
    print(f"\n{'='*60}")
    print(f"BENCHMARKING: {backend.upper()}")
    print(f"{'='*60}")
    
    # Backend descriptions
    backend_desc = {
        'csr': "CSR (Compressed Sparse Row) - efficient for row-wise access",
        'coo': "COO (Coordinate) - flexible for irregular sparsity",
        'bsr': f"BSR (Block Sparse Row {block_size}x{block_size}) - efficient for block patterns",
        'hybrid': "Hybrid (dense matmul on thresholded matrix) - vectorized operations",
    }
    print(f"Strategy: {backend_desc.get(backend, 'Unknown')}")
    
    # Warmup runs
    print("Warming up...")
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
    seq_q = attention_weights.shape[-2]
    seq_k = attention_weights.shape[-1]
    head_dim = attention_V.shape[-1]
    leading = int(np.prod(attention_weights.shape[:-2])) if attention_weights.dim() > 2 else 1
    return float(2 * leading * seq_q * seq_k * head_dim)

def print_benchmark_results(times, label, attention_weights, attention_V):
    """Print benchmark statistics."""
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    median_time = np.median(times)
    
    # Percentiles
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
    
    # Compute throughput
    flops = _equivalent_flops(attention_weights, attention_V)
    throughput_mean = flops / mean_time / 1e9
    throughput_trimmed = flops / trimmed_mean / 1e9
    print(f"  Throughput:      {throughput_mean:.2f} GFLOPS (mean), {throughput_trimmed:.2f} GFLOPS (trimmed)")
    
    return trimmed_mean

# ============================================================================
# Run all benchmarks
# ============================================================================

# List of backends to compare
backends_to_compare = ['hybrid', 'csr', 'coo', 'bsr']

# List of reordering methods to compare (if enabled)
reordering_methods = ['cache_cluster', 'degree', 'centroid', 'profile']

results = {}

# 1. Benchmark DENSE (baseline)
print(f"\n{'#'*60}")
print(f"# STEP 1: DENSE BASELINE")
print(f"{'#'*60}")
dense_times, dense_output = benchmark_dense(
    attention_weights, attention_V, num_warmup, num_iterations, device
)
results["dense"] = {
    "times": dense_times,
    "output": dense_output,
    "mean": print_benchmark_results(dense_times, "DENSE", attention_weights, attention_V)
}

# 2. Benchmark ALL SPARSE BACKENDS (without reordering)
print(f"\n{'#'*60}")
print(f"# STEP 2: SPARSE BACKENDS COMPARISON (baseline)")
print(f"{'#'*60}")

for backend in backends_to_compare:
    try:
        sparse_times, sparse_output = benchmark_sparse(
            attention_weights, attention_V, num_warmup, num_iterations, device, EPS,
            backend=backend, block_size=args.block_size
        )
        results[backend] = {
            "times": sparse_times,
            "output": sparse_output,
            "mean": print_benchmark_results(sparse_times, backend.upper(), attention_weights, attention_V)
        }
    except Exception as e:
        print(f"\n⚠️  Backend {backend.upper()} failed: {e}")
        results[backend] = None

# 3. Benchmark SPARSE BACKENDS WITH REORDERING (if enabled)
if args.include_reordering:
    print(f"\n{'#'*60}")
    print(f"# STEP 3: SPARSE BACKENDS WITH REORDERING")
    print(f"{'#'*60}")
    print(f"Reordering axis: {args.reorder}")
    
    # Compute mask once for all reordering methods
    print(f"\nComputing sparsity mask (eps={EPS})...")
    mask2d = _aggregate_mask_2d(attention_weights, EPS)
    print(f"Aggregated mask shape: {tuple(mask2d.shape)}")
    
    # Analyze baseline span
    span_stats_baseline = _row_span_stats(mask2d)
    print(f"\nBaseline row span statistics:")
    print(f"  Avg: {span_stats_baseline['avg']:.2f} | Median: {span_stats_baseline['median']:.2f} | " +
          f"Max: {span_stats_baseline['max']} | P95: {span_stats_baseline['p95']:.2f}")
    
    for reorder_method in reordering_methods:
        print(f"\n{'='*60}")
        print(f"REORDERING METHOD: {reorder_method.upper()}")
        print(f"{'='*60}")
        
        # Compute reordering
        print(f"Computing reordering (method={reorder_method})...")
        try:
            row_perm, col_perm = compute_reordering_from_mask(mask2d, method=reorder_method)
            
            # Analyze reordered span
            mask2d_reordered = mask2d.index_select(0, row_perm).index_select(1, col_perm)
            span_stats_reordered = _row_span_stats(mask2d_reordered)
            
            avg_improvement = (span_stats_baseline['avg'] - span_stats_reordered['avg']) / span_stats_baseline['avg'] * 100 if span_stats_baseline['avg'] > 0 else 0
            max_improvement = (span_stats_baseline['max'] - span_stats_reordered['max']) / span_stats_baseline['max'] * 100 if span_stats_baseline['max'] > 0 else 0
            
            print(f"Row span after reordering:")
            print(f"  Avg: {span_stats_reordered['avg']:.2f} | Median: {span_stats_reordered['median']:.2f} | " +
                  f"Max: {span_stats_reordered['max']} | P95: {span_stats_reordered['p95']:.2f}")
            print(f"  Improvement: Avg span reduced by {avg_improvement:.1f}%, Max span reduced by {max_improvement:.1f}%")
            
            # Apply reordering to attention and V
            attn_r, V_r, inv_row_perm = apply_reordering(attention_weights, attention_V, row_perm, col_perm, args.reorder)
            
            # Benchmark each backend with this reordering
            for backend in backends_to_compare:
                if results.get(backend) is None:
                    continue
                
                result_key = f"{backend}_{reorder_method}"
                
                try:
                    print(f"\nBenchmarking {backend.upper()} with {reorder_method} reordering...")
                    
                    # Warmup
                    for _ in range(num_warmup):
                        out_r = sparse_attention_matmul(attn_r, V_r, EPS, device, backend, args.block_size)
                        if inv_row_perm is not None:
                            out_r = out_r.index_select(dim=-2, index=inv_row_perm)
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                    
                    # Benchmark
                    times = []
                    for _ in range(num_iterations):
                        start_time = time.perf_counter()
                        out_r = sparse_attention_matmul(attn_r, V_r, EPS, device, backend, args.block_size)
                        if inv_row_perm is not None:
                            out_r = out_r.index_select(dim=-2, index=inv_row_perm)
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        end_time = time.perf_counter()
                        times.append(end_time - start_time)
                    
                    sparse_r_times = np.array(times)
                    results[result_key] = {
                        "times": sparse_r_times,
                        "output": out_r,
                        "mean": print_benchmark_results(
                            sparse_r_times, 
                            f"{backend.upper()} + {reorder_method.upper()}", 
                            attention_weights, attention_V
                        ),
                        "reorder_method": reorder_method,
                        "backend": backend
                    }
                    
                except Exception as e:
                    print(f"⚠️  {backend.upper()} with {reorder_method} failed: {e}")
                    results[result_key] = None
        
        except Exception as e:
            print(f"⚠️  Reordering method {reorder_method} failed: {e}")
            continue

# ============================================================================
# Comparison Summary
# ============================================================================

print(f"\n{'='*60}")
print(f"COMPREHENSIVE COMPARISON SUMMARY")
print(f"{'='*60}")

dense_mean = results["dense"]["mean"]
print(f"\nDense (baseline):    {dense_mean*1000:.4f} ms")
print(f"\nSparse backends (no reordering):")

# Collect valid baseline results
valid_backends = []
for backend in backends_to_compare:
    if results.get(backend) is not None:
        sparse_mean = results[backend]["mean"]
        speedup = dense_mean / sparse_mean
        valid_backends.append((backend, sparse_mean, speedup))
        
        if speedup >= 1:
            print(f"  {backend.upper():12s} {sparse_mean*1000:8.4f} ms  |  Speedup: {speedup:5.2f}x 🚀")
        else:
            print(f"  {backend.upper():12s} {sparse_mean*1000:8.4f} ms  |  Slowdown: {1/speedup:5.2f}x ⚠️")

# Find fastest sparse backend (baseline)
fastest_baseline = None
if valid_backends:
    fastest_backend, fastest_time, fastest_speedup = min(valid_backends, key=lambda x: x[1])
    fastest_baseline = (fastest_backend, fastest_time, fastest_speedup)
    print(f"\n🏆 FASTEST SPARSE BACKEND (no reordering): {fastest_backend.upper()}")
    print(f"   Time: {fastest_time*1000:.4f} ms")
    print(f"   Speedup vs Dense: {fastest_speedup:.2f}x")
    
    # Compare sparse backends against each other
    print(f"\n{'='*60}")
    print(f"SPARSE BACKEND COMPARISON (relative to fastest)")
    print(f"{'='*60}")
    for backend, backend_time, _ in sorted(valid_backends, key=lambda x: x[1]):
        relative_speed = fastest_time / backend_time
        if backend == fastest_backend:
            print(f"  {backend.upper():12s} {backend_time*1000:8.4f} ms  |  1.00x (fastest) 🏆")
        else:
            print(f"  {backend.upper():12s} {backend_time*1000:8.4f} ms  |  {1/relative_speed:.2f}x slower")

# REORDERING RESULTS
if args.include_reordering:
    print(f"\n{'='*60}")
    print(f"REORDERING RESULTS (axis: {args.reorder})")
    print(f"{'='*60}")
    
    # Collect all reordered results
    reordered_results = []
    for key, result in results.items():
        if result is not None and "_" in key and key not in backends_to_compare:
            backend = result.get("backend", key.split("_")[0])
            reorder_method = result.get("reorder_method", "_".join(key.split("_")[1:]))
            reordered_results.append((key, backend, reorder_method, result["mean"]))
    
    if reordered_results:
        # Group by reordering method
        by_method = {}
        for key, backend, method, mean_time in reordered_results:
            if method not in by_method:
                by_method[method] = []
            by_method[method].append((backend, mean_time, key))
        
        # Print results by reordering method
        for method in reordering_methods:
            if method not in by_method:
                continue
            
            print(f"\n{method.upper()} reordering:")
            method_results = sorted(by_method[method], key=lambda x: x[1])
            
            for backend, mean_time, key in method_results:
                speedup_vs_dense = dense_mean / mean_time
                
                # Compare to baseline (non-reordered) version
                baseline_time = results.get(backend, {}).get("mean")
                if baseline_time:
                    improvement = (baseline_time - mean_time) / baseline_time * 100
                    if improvement > 0:
                        print(f"  {backend.upper():12s} {mean_time*1000:8.4f} ms  |  vs dense: {speedup_vs_dense:5.2f}x  |  vs baseline: +{improvement:5.1f}% faster ⚡")
                    else:
                        print(f"  {backend.upper():12s} {mean_time*1000:8.4f} ms  |  vs dense: {speedup_vs_dense:5.2f}x  |  vs baseline: {improvement:5.1f}% slower ⚠️")
                else:
                    print(f"  {backend.upper():12s} {mean_time*1000:8.4f} ms  |  vs dense: {speedup_vs_dense:5.2f}x")
        
        # Find OVERALL fastest (including reordering)
        all_results = [(backend, mean_time, None) for backend, mean_time, _ in valid_backends]
        all_results.extend([(f"{backend}+{method}", mean_time, method) for _, backend, method, mean_time in reordered_results])
        
        fastest_name, fastest_time_overall, fastest_method = min(all_results, key=lambda x: x[1])
        speedup_overall = dense_mean / fastest_time_overall
        
        print(f"\n{'='*60}")
        print(f"🏆 OVERALL FASTEST CONFIGURATION")
        print(f"{'='*60}")
        print(f"  Config: {fastest_name.upper()}")
        print(f"  Time: {fastest_time_overall*1000:.4f} ms")
        print(f"  Speedup vs Dense: {speedup_overall:.2f}x")
        
        if fastest_baseline:
            baseline_name, baseline_time, _ = fastest_baseline
            improvement_vs_baseline = (baseline_time - fastest_time_overall) / baseline_time * 100
            if improvement_vs_baseline > 0:
                print(f"  Improvement vs best baseline ({baseline_name.upper()}): {improvement_vs_baseline:.1f}% faster")
            else:
                print(f"  vs best baseline ({baseline_name.upper()}): {-improvement_vs_baseline:.1f}% slower")
        
        # Best reordering method per backend
        print(f"\n{'='*60}")
        print(f"BEST REORDERING METHOD PER BACKEND")
        print(f"{'='*60}")
        
        for backend in backends_to_compare:
            if results.get(backend) is None:
                continue
            
            baseline_time = results[backend]["mean"]
            backend_reordered = [(method, mean_time, key) for key, b, method, mean_time in reordered_results if b == backend]
            
            if backend_reordered:
                best_method, best_time, best_key = min(backend_reordered, key=lambda x: x[1])
                improvement = (baseline_time - best_time) / baseline_time * 100
                
                if improvement > 0:
                    print(f"  {backend.upper():12s} baseline: {baseline_time*1000:7.4f} ms  →  {best_method}: {best_time*1000:7.4f} ms  (+{improvement:5.1f}% faster) ⚡")
                else:
                    print(f"  {backend.upper():12s} baseline: {baseline_time*1000:7.4f} ms  →  {best_method}: {best_time*1000:7.4f} ms  ({improvement:5.1f}% slower) ⚠️")
            else:
                print(f"  {backend.upper():12s} baseline: {baseline_time*1000:7.4f} ms  (no reordering tested)")
    else:
        print("  No reordered results available.")

# ============================================================================
# Output Accuracy Check
# ============================================================================

print(f"\n{'='*60}")
print(f"OUTPUT ACCURACY (vs DENSE baseline)")
print(f"{'='*60}")

dense_out = results["dense"]["output"]

for backend in backends_to_compare:
    if results.get(backend) is None:
        continue
    
    sparse_out = results[backend]["output"]
    abs_diff = torch.abs(dense_out - sparse_out)
    mae = torch.mean(abs_diff).item()
    max_diff = torch.max(abs_diff).item()
    mse = torch.mean((dense_out - sparse_out) ** 2).item()
    rmse = float(np.sqrt(mse))
    dense_mean_abs = torch.mean(torch.abs(dense_out)).item()
    relative_mae = mae / (dense_mean_abs + 1e-10)
    
    print(f"\n{backend.upper()}:")
    print(f"  MAE:          {mae:.6e}  |  Relative: {relative_mae*100:.4f}%")
    print(f"  Max diff:     {max_diff:.6e}")
    print(f"  RMSE:         {rmse:.6e}")

print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"✅ Benchmark complete!")
print(f"   Sparsity: {sparsity*100:.2f}% (eps={EPS})")
print(f"   Avg non-zeros per row: {avg_nnz_per_row:.2f}")
print(f"   Device: {device}")
print(f"   Iterations: {num_iterations} (warmup: {num_warmup})")
print(f"   Dense baseline: {dense_mean*1000:.4f} ms")

if valid_backends:
    fastest_backend, fastest_time, fastest_speedup = min(valid_backends, key=lambda x: x[1])
    print(f"   Best sparse backend (no reordering): {fastest_backend.upper()} ({fastest_speedup:.2f}x speedup)")

if args.include_reordering:
    # Find overall fastest including reordering
    all_times = []
    for backend in backends_to_compare:
        if results.get(backend) is not None:
            all_times.append((backend, results[backend]["mean"], None))
    
    for key, result in results.items():
        if result is not None and "_" in key and key not in backends_to_compare:
            backend = result.get("backend", key.split("_")[0])
            method = result.get("reorder_method", "_".join(key.split("_")[1:]))
            all_times.append((f"{backend}+{method}", result["mean"], method))
    
    if all_times:
        fastest_overall_name, fastest_overall_time, _ = min(all_times, key=lambda x: x[1])
        fastest_overall_speedup = dense_mean / fastest_overall_time
        print(f"   BEST OVERALL: {fastest_overall_name.upper()} ({fastest_overall_speedup:.2f}x speedup, {fastest_overall_time*1000:.4f} ms)")

print(f"\n💡 TIP: To include reordering comparison, add --include-reordering flag")
print(f"{'='*60}")


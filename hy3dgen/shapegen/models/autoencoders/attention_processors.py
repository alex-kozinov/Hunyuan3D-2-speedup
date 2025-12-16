# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import math
import os

import torch
import torch.nn.functional as F
from torch import vmap

def _manual_scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    d_k = q.size(-1)
    scale = 1.0 / math.sqrt(d_k)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_probs = torch.softmax(attn_scores, dim=-1)
    return torch.matmul(attn_probs, v)


def _batched_sparse_mm(attn_probs: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    attn_probs: [B, H, Lq, Lk] (dense, уже с занулёнными мелкими значениями)
    v:          [B, H, Lk, Dv]
    return:     [B, H, Lq, Dv]
    """
    b, h, l_q, l_k = attn_probs.shape
    d_v = v.shape[-1]
    orig_dtype = v.dtype

    attn_flat = attn_probs.reshape(b * h, l_q, l_k)
    v_flat    = v.reshape(b * h, l_k, d_v)

    outs = []
    for idx in range(b * h):
        attn_single = attn_flat[idx].to(torch.float32)
        v_single    = v_flat[idx].to(torch.float32)

        attn_sparse = attn_single.to_sparse()
        out_single  = torch.sparse.mm(attn_sparse, v_single)  # [Lq, Dv]

        outs.append(out_single.to(orig_dtype))

    out = torch.stack(outs, dim=0).reshape(b, h, l_q, d_v)
    return out


def _sparse_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sparsity_threshold: float = 1e-4,
) -> torch.Tensor:
    """
    q, k, v: [B, H, L, D]
    """
    b, h, l_q, d_k = q.shape
    _, _, l_k, d_v = v.shape
    scale = 1.0 / math.sqrt(d_k)

    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B,H,Lq,Lk]
    attn_probs = torch.softmax(attn_scores, dim=-1)

    if sparsity_threshold is not None:
        attn_probs = torch.where(
            attn_probs >= sparsity_threshold,
            attn_probs,
            torch.zeros_like(attn_probs),
        )

    out = _batched_sparse_mm(attn_probs, v)
    return out

scaled_dot_product_attention = _manual_scaled_dot_product_attention
if os.environ.get('CA_USE_SAGEATTN', '0') == '1':
    try:
        from sageattention import sageattn
    except ImportError:
        raise ImportError('Please install the package "sageattention" to use this USE_SAGEATTN.')
    scaled_dot_product_attention = sageattn


class CrossAttentionProcessor:
    def __call__(self, attn, q, k, v):
        out = _manual_scaled_dot_product_attention(q, k, v)
        return out

class SparseFlashCrossAttentionProcessor:
    def __call__(self, attn, q, k, v):
        out = _sparse_scaled_dot_product_attention(q, k, v)
        return out


class FlashVDMCrossAttentionProcessor:
    def __init__(self, topk=None):
        self.topk = topk

    def __call__(self, attn, q, k, v):
        if k.shape[-2] == 3072:
            topk = 1024
        elif k.shape[-2] == 512:
            topk = 256
        else:
            topk = k.shape[-2] // 3

        if self.topk is True:
            q1 = q[:, :, ::100, :]
            sim = q1 @ k.transpose(-1, -2)
            sim = torch.mean(sim, -2)
            topk_ind = torch.topk(sim, dim=-1, k=topk).indices.squeeze(-2).unsqueeze(-1)
            topk_ind = topk_ind.expand(-1, -1, -1, v.shape[-1])
            v0 = torch.gather(v, dim=-2, index=topk_ind)
            k0 = torch.gather(k, dim=-2, index=topk_ind)
            out = scaled_dot_product_attention(q, k0, v0)
        elif self.topk is False:
            out = scaled_dot_product_attention(q, k, v)
        else:
            idx, counts = self.topk
            start = 0
            outs = []
            for grid_coord, count in zip(idx, counts):
                end = start + count
                q_chunk = q[:, :, start:end, :]
                k0, v0 = self.select_topkv(q_chunk, k, v, topk)
                out = scaled_dot_product_attention(q_chunk, k0, v0)
                outs.append(out)
                start += count
            out = torch.cat(outs, dim=-2)
        self.topk = False
        return out

    def select_topkv(self, q_chunk, k, v, topk):
        q1 = q_chunk[:, :, ::50, :]
        sim = q1 @ k.transpose(-1, -2)
        sim = torch.mean(sim, -2)
        topk_ind = torch.topk(sim, dim=-1, k=topk).indices.squeeze(-2).unsqueeze(-1)
        topk_ind = topk_ind.expand(-1, -1, -1, v.shape[-1])
        v0 = torch.gather(v, dim=-2, index=topk_ind)
        k0 = torch.gather(k, dim=-2, index=topk_ind)
        return k0, v0


class FlashVDMTopMCrossAttentionProcessor(FlashVDMCrossAttentionProcessor):
    def select_topkv(self, q_chunk, k, v, topk):
        q1 = q_chunk[:, :, ::30, :]
        sim = q1 @ k.transpose(-1, -2)
        # sim = sim.to(torch.float32)
        sim = sim.softmax(-1)
        sim = torch.mean(sim, 1)
        activated_token = torch.where(sim > 1e-6)[2]
        index = torch.unique(activated_token, return_counts=True)[0].unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        index = index.expand(-1, v.shape[1], -1, v.shape[-1])
        v0 = torch.gather(v, dim=-2, index=index)
        k0 = torch.gather(k, dim=-2, index=index)
        return k0, v0

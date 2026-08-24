import math

import torch
import torch.nn.functional as F


def _as_valid_mask(attention_mask):
    if attention_mask.dtype == torch.bool:
        # Heuristic for compatibility:
        # - flash_attn-style masks usually mark valid tokens as True
        # - PyTorch key_padding_mask usually marks padded tokens as True
        if attention_mask.sum() <= attention_mask.numel() / 2:
            return ~attention_mask
        return attention_mask
    return attention_mask > 0


def index_first_axis(x, indices):
    return x[indices]


def unpad_input(x, attention_mask):
    valid_mask = _as_valid_mask(attention_mask).to(device=x.device)
    batch_size, seqlen = valid_mask.shape
    flat_mask = valid_mask.reshape(batch_size * seqlen)
    indices = flat_mask.nonzero(as_tuple=False).squeeze(-1)
    x_unpad = index_first_axis(x.reshape(batch_size * seqlen, *x.shape[2:]), indices)
    seqlens = valid_mask.sum(dim=1, dtype=torch.int32)
    cu_seqlens = torch.zeros(
        seqlens.numel() + 1, dtype=torch.int32, device=x.device
    )
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
    max_seqlen = int(seqlens.max().item()) if seqlens.numel() > 0 else 0
    return x_unpad, indices, cu_seqlens, max_seqlen


def pad_input(x_unpad, indices, batch_size, seqlen):
    output = x_unpad.new_zeros((batch_size * seqlen,) + x_unpad.shape[1:])
    output[indices] = x_unpad
    return output.reshape(batch_size, seqlen, *x_unpad.shape[1:])


def _causal_mask(query_len, key_len, device):
    diagonal = max(key_len - query_len + 1, 1)
    return torch.triu(
        torch.ones(query_len, key_len, dtype=torch.bool, device=device),
        diagonal=diagonal,
    )


def _scaled_dot_product_attention(q, k, v, dropout_p, softmax_scale, causal):
    q = q.transpose(0, 1)  # [H, Lq, D]
    k = k.transpose(0, 1)  # [H, Lk, D]
    v = v.transpose(0, 1)  # [H, Lk, D]

    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        attn = attn.masked_fill(_causal_mask(q.shape[1], k.shape[1], q.device), float("-inf"))
    attn = torch.softmax(attn.float(), dim=-1).to(q.dtype)
    if dropout_p > 0:
        attn = F.dropout(attn, p=dropout_p, training=True)
    output = torch.matmul(attn, v)
    return output.transpose(0, 1).contiguous()


def flash_attn_unpadded_kvpacked_func(
    q,
    kv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale=None,
    causal=False,
):
    del max_seqlen_q, max_seqlen_k

    outputs = []
    batch_size = cu_seqlens_q.numel() - 1
    for batch_idx in range(batch_size):
        q_start = int(cu_seqlens_q[batch_idx].item())
        q_end = int(cu_seqlens_q[batch_idx + 1].item())
        k_start = int(cu_seqlens_k[batch_idx].item())
        k_end = int(cu_seqlens_k[batch_idx + 1].item())

        q_i = q[q_start:q_end]
        kv_i = kv[k_start:k_end]
        k_i = kv_i[:, 0]
        v_i = kv_i[:, 1]
        outputs.append(
            _scaled_dot_product_attention(
                q_i, k_i, v_i, dropout_p, softmax_scale, causal
            )
        )

    if outputs:
        return torch.cat(outputs, dim=0)
    return q.new_empty((0,) + q.shape[1:])


def flash_attn_varlen_kvpacked_func(*args, **kwargs):
    return flash_attn_unpadded_kvpacked_func(*args, **kwargs)

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    MLP block with GELU activation and dropout.
    """
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module with optional fused attention support.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1, fused_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.fused_attn = fused_attn
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, out_proj):
        B, T, D = Q.shape
        head_dim = D // self.num_heads

        Q_ = Q.view(B, T, self.num_heads, head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        K_ = K.view(B, -1, self.num_heads, head_dim).transpose(1, 2)
        V_ = V.view(B, -1, self.num_heads, head_dim).transpose(1, 2)

        if self.fused_attn:
            context = F.scaled_dot_product_attention(
                Q_, K_, V_,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False
            )
        else:
            scores = torch.matmul(Q_, K_.transpose(-1, -2)) / (head_dim ** 0.5)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            context = torch.matmul(attn_weights, V_)  # (B, num_heads, T, head_dim)

        context = context.transpose(1, 2).contiguous().view(B, T, D)
        out = out_proj(context)
        return out

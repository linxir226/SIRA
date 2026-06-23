"""Lightweight query transformer (QTrans) used by SIRA."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class VisionMlp(nn.Module):
    def __init__(self, dim, hidden_dim, hidden_act="quick_gelu"):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = QuickGELU() if hidden_act == "quick_gelu" else nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class VisionSdpaAttentionSimple(nn.Module):
    def __init__(self, dim, num_heads=16):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"embed_dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, hidden_states, cu_seqlens=None, rotary_pos_emb=None, position_embeddings=None):
        del cu_seqlens, rotary_pos_emb, position_embeddings
        batch_size, seq_len, channels = hidden_states.shape
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(batch_size, seq_len, 3, self.num_heads, -1)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, channels)
        return self.proj(output)


class QueryTransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim=1280,
        num_heads=16,
        mlp_ratio=4,
        hidden_act="quick_gelu",
        attn_implementation="sdpa",
    ):
        super().__init__()
        if attn_implementation != "sdpa":
            raise ValueError("QueryTransformerBlock currently supports only SDPA attention")
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = VisionSdpaAttentionSimple(embed_dim, num_heads)
        self.mlp = VisionMlp(
            dim=embed_dim,
            hidden_dim=int(embed_dim * mlp_ratio),
            hidden_act=hidden_act,
        )

    def forward(self, hidden_states, cu_seqlens=None, rotary_pos_emb=None, position_embeddings=None):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class QueryTransformer(nn.Module):
    def __init__(
        self,
        depth,
        seq_len,
        embed_dim=256,
        num_heads=16,
        mlp_ratio=4,
        hidden_act="quick_gelu",
        attn_implementation="sdpa",
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                QueryTransformerBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    hidden_act,
                    attn_implementation,
                )
                for _ in range(depth)
            ]
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))

    def forward(self, hidden_states, cu_seqlens=None, rotary_pos_emb=None):
        if hidden_states.ndim != 3:
            raise ValueError(
                f"Expected hidden states with shape [B, L, C], got {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[1:] != self.pos_embed.shape[1:]:
            raise ValueError(
                "Query transformer input shape does not match positional embedding: "
                f"{tuple(hidden_states.shape[1:])} vs {tuple(self.pos_embed.shape[1:])}"
            )

        hidden_states = hidden_states + self.pos_embed
        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, rotary_pos_emb)
        return hidden_states


__all__ = ["QueryTransformer"]

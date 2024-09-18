import torch
import torch.nn as nn

from .feedforward import SwiGLU


class TransformerEncoderBlock(nn.Module):
    def __init__(self, ndim, nheads, dropout):
        super().__init__()
        self.nheads = nheads
        self.norm1 = nn.LayerNorm(ndim)
        self.attn = nn.MultiheadAttention(
            ndim, nheads, dropout=dropout, batch_first=True
        )

        self.norm2 = nn.LayerNorm(ndim)
        self.ff = SwiGLU(ndim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None, mask_type="src", need_weights=False):
        b, seq_len = x.size()[:2]
        # x (b * seq_len, npatch, ndim)
        if mask is not None:
            if mask_type == "src":
                mask = create_src_mask(mask, b, seq_len, self.nheads)
            elif mask_type == "tgt":
                mask = create_tgt_mask(mask, b, seq_len, self.nheads)
            else:
                raise ValueError
        x = self.norm1(x)
        x_attn, attn_w = self.attention_block(x, mask, need_weights)
        x = x + x_attn

        x = self.norm2(x)
        x = x + self.feed_forward_block(x)

        return x, attn_w

    def attention_block(self, x, mask, need_weights):
        x, attn_w = self.attn(x, x, x, attn_mask=mask, need_weights=need_weights)
        return x, attn_w

    def feed_forward_block(self, x):
        x = self.ff(x)
        x = self.dropout2(x)
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, ndim, nheads, dropout):
        super().__init__()
        self.nheads = nheads
        self.norm1 = nn.LayerNorm(ndim)
        self.attn1 = nn.MultiheadAttention(
            ndim, nheads, dropout=dropout, batch_first=True
        )

        self.norm2 = nn.LayerNorm(ndim)
        self.attn2 = nn.MultiheadAttention(
            ndim, nheads, dropout=dropout, batch_first=True
        )

        self.norm3 = nn.LayerNorm(ndim)
        self.ff = SwiGLU(ndim)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, z, mask=None):
        b, seq_len = x.size()[:2]
        # x (b * seq_len, npatch, ndim)
        tgt_mask = create_tgt_mask(mask, b, seq_len, self.nheads, x.device)
        x = self.norm1(x)
        x = x + self.attention_block1(x, tgt_mask)

        if mask is not None:
            src_mask = create_src_mask(mask, b, seq_len, self.nheads)
        else:
            src_mask = None
        x = self.norm2(x)
        x = x + self.attention_block2(x, z, src_mask)

        x = self.norm3(x)
        x = x + self.feed_forward_block(x)

        return x

    def attention_block1(self, x, mask):
        x = self.attn1(x, x, x, attn_mask=mask, need_weights=False)[0]
        return x

    def attention_block2(self, x, z, mask):
        x = self.attn2(x, z, z, attn_mask=mask, need_weights=False)[0]
        return x

    def feed_forward_block(self, x):
        x = self.ff(x)
        x = self.dropout3(x)
        return x


def create_src_mask(mask, b, seq_len, nheads):
    mask = (
        mask.view(b, 1, seq_len)
        .repeat((1, nheads, seq_len))
        .view(b * nheads, seq_len, seq_len)
    ).detach()
    return mask


def create_tgt_mask(mask, b, seq_len, nheads, device=None):
    if mask is not None:
        mask = create_src_mask(mask, b, seq_len, nheads)
        subsequent_mask = ~torch.tril(torch.full((seq_len, seq_len), True)).to(mask.device)
        mask = mask + subsequent_mask
    else:
        mask = ~torch.tril(torch.full((seq_len, seq_len), True)).to(device)
    return mask

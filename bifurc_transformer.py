import torch
from torch import nn
import numpy as np


class PosEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, pos_encoding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_encoding = pos_encoding
    def forward(self, src, pos):
        src = self.pos_encoding(src, pos)
        return super().forward(src)


class PosEncoder(nn.TransformerEncoder):
    def forward(self, src, pos):
        output = src
        for mod in self.layers:
            output = mod(output, pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class PosDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, pos_encoding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_encoding = pos_encoding

    def forward(self, tgt, memory, queries, pos):
        x = tgt
        x = self.norm1(x + self._sa_block(x, queries))
        x = self.norm2(x + self._mha_block(x, memory, queries, pos))
        x = self.norm3(x + self._ff_block(x))
        return x

    # self-attention block
    def _sa_block(self, x, queries):
        xq = x + queries
        x = self.self_attn(xq, xq, x, need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x, mem, queries, pos):
        mem_pos = self.pos_encoding(mem, pos)
        xq = x + queries
        x = self.multihead_attn(xq, mem_pos, mem, need_weights=False)[0]
        return self.dropout2(x)


class PosDecoder(nn.TransformerDecoder):
    def forward(self, tgt, memory, pos):
        output = tgt
        for mod in self.layers:
            output = mod(output, memory, tgt, pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class PositionalEncoding(nn.Module):
    # modified from pytorch.org tutorials
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        # self.dropout = nn.Dropout(p=dropout)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(4.) / d_model))
        self.register_buffer('div_term', div_term.cuda())

    def forward(self, x, r):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        pe = torch.zeros_like(x)
        pe[..., 0::2] = torch.sin(r[..., None] * self.div_term)
        pe[..., 1::2] = torch.cos(r[..., None] * self.div_term)
        # return self.dropout(x)
        return x + pe

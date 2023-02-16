import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Tuple
import numpy as np


class MultiHeadSelfAttention(nn.Module):
    @staticmethod
    def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   d_k: int = None,
                   mask: torch.tensor = None, dropout: float = None):
        if d_k is None:
            d_k = k.shape[-1]
        # transpose k's last two dimensions
        k = torch.transpose(k, -1, -2)
        attn = torch.matmul(q / math.sqrt(d_k), k)

        if mask is not None:
            attn = torch.masked_fill(attn, mask, 1e-9)
        attn = F.softmax(attn, dim=-1)

        if dropout is not None:
            attn = F.dropout(attn, dropout)

        out = torch.matmul(attn, v)

        return out, attn

    linear_q: nn.Linear
    linear_k: nn.Linear
    linear_v: nn.Linear
    fc: nn.Linear
    dropout: float
    dropper: nn.Module
    norm: nn.LayerNorm

    head: int
    d_k: int
    d_v: int

    def __init__(self, d_emb: int, head: int, d_k: int,
                 d_v: int = None,
                 dropout: float = 0.1,
                 bias_qkv: bool = False):
        super().__init__()
        if d_v is None:
            d_v = d_k
        self.linear_q = nn.Linear(d_emb, head * d_k, bias=bias_qkv)
        self.linear_k = nn.Linear(d_emb, head * d_k, bias=bias_qkv)
        self.linear_v = nn.Linear(d_emb, head * d_v, bias=bias_qkv)
        self.fc = nn.Linear(head * d_v, d_emb, bias=bias_qkv)
        self.dropout = dropout

        if dropout is not None:
            self.dropper = nn.Dropout(dropout)
        else:
            self.dropper = nn.Identity()

        self.norm = nn.LayerNorm(d_emb, eps=1e-6)

        self.head = head
        self.d_k = d_k
        self.d_v = d_v

    def forward(self, q: torch.Tensor, k: torch.Tensor = None, v: torch.Tensor = None,
                mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward for transformer
        :param q: query, tensor(N, *, d_emb)
        :param k: key (default q), tensor(N, *, d_emb)
        :param v: value (default v), tensor(N, *, d_emb)
        :param mask: mask, tensor(sequence)
        :return: weight, attention
        """
        resid = q
        if k is None:
            k = q
        if v is None:
            v = q
        if mask is not None:
            mask = torch.unsqueeze(mask, -3)  # mask on the expected head dim

        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)  # (N, S, H * S)
        q = torch.transpose(q.view(*q.shape[:-1], self.head, self.d_k), -2, -3)  # (N, H, S, S)
        k = torch.transpose(k.view(*k.shape[:-1], self.head, self.d_k), -2, -3)  # (N, H, S, S)
        v = torch.transpose(v.view(*v.shape[:-1], self.head, self.d_v), -2, -3)  # (N, H, S, S)

        weight, attn = self._attention(q, k, v, d_k=self.d_k, mask=mask, dropout=self.dropout)
        weight = weight.transpose(-2, -3)
        weight = weight.contiguous().view(*weight.shape[:-2], -1)
        weight = self.dropper(self.fc(weight))

        weight += resid

        weight = self.norm(weight)

        return weight, attn


class FeedForward(nn.Module):
    dropout: float
    w: nn.Module
    norm: nn.Module

    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.w = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        resid = x
        x = self.w(x)
        x += resid
        x = self.norm(x)
        return x


class TransformerEncodingLayer(nn.Module):
    attention: nn.Module
    feed_forward: nn.Module

    def __init__(self, d_emb: int,
                 head: int = 1,
                 d_k: int = 1024, d_v: int = 1024,
                 attention_dropout: float = 0.1,
                 forward_hidden: int = 1024,
                 forward_dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_emb, head, d_k, d_v, dropout=attention_dropout)
        self.feed_forward = FeedForward(d_emb, forward_hidden, forward_dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x, _ = self.attention(x, mask=mask)
        x = self.feed_forward(x)
        return x


class TransformerEncoder(nn.Module):
    layer: int
    layers: nn.ModuleList
    norm: nn.Module

    def __init__(self, layer: int, d_emb: int,
                 head: int = 1, d_k=1024, d_v=1024,
                 attention_dropout: float = 0.1,
                 forward_hidden: int = 1024, forward_dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncodingLayer(d_emb, head, d_k, d_v, attention_dropout, forward_hidden, forward_dropout)
             for _ in range(layer)]
        )
        self.layer = layer
        self.norm = nn.LayerNorm(d_emb, eps=1e-6)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


class TransformerDecodingLayer(nn.Module):
    attention: nn.Module
    encoder_decoder_attention: nn.Module
    feed_forward: nn.Module

    def __init__(self, d_emb: int,
                 head: int = 1,
                 d_k: int = 1024, d_v: int = 1024,
                 attention_dropout: float = 0.1,
                 forward_hidden: int = 1024,
                 forward_dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_emb, head, d_k, d_v, dropout=attention_dropout)
        self.encoder_decoder_attention = MultiHeadSelfAttention(d_emb, head, d_k, d_v, dropout=attention_dropout)
        self.feed_forward = FeedForward(d_emb, forward_hidden, forward_dropout)

    def forward(self, x: torch.Tensor,
                encoder_kv: torch.Tensor,
                mask_encoder: torch.Tensor,
                mask_decoder: torch.Tensor) -> torch.Tensor:
        """
        :param x: [N, seq, dim]
        :param encoder_kv: [N, seq, dim]
        :param mask_encoder: [N, seq, seq]
        :param mask_decoder: [N, seq, seq]
        :return:
        """
        x, _ = self.attention(x, mask=mask_decoder)
        # Decoder in transformer depends on encoder's output
        x, _ = self.attention(x, encoder_kv, encoder_kv, mask=mask_encoder)
        x = self.feed_forward(x)
        return x


class TransformerDecoder(nn.Module):
    layer: int
    layers: nn.ModuleList
    norm: nn.Module

    def __init__(self, layer: int, d_emb: int,
                 head: int = 1, d_k=1024, d_v=1024,
                 attention_dropout: float = 0.1,
                 forward_hidden: int = 1024, forward_dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerDecodingLayer(d_emb, head, d_k, d_v, attention_dropout, forward_hidden, forward_dropout)
                for _ in range(layer)
            ]
        )
        self.layer = layer
        self.norm = nn.LayerNorm(d_emb, eps=1e-6)

    def forward(self, x: torch.Tensor,
                encoder_output: torch.Tensor,
                mask_encoder: torch.Tensor,
                mask_decoder: torch.Tensor):
        """
        :param x: [N, seq, dim]
        :param encoder_output: [N, seq, dim]
        :param mask_encoder: [N, seq, seq]
        :param mask_decoder: [N, seq, seq]
        :return:
        """
        for layer in self.layers:
            x = layer(x, encoder_output, mask_encoder, mask_decoder)
        return x


class PositionalEncoding(nn.Module):
    pos: torch.Tensor

    def __init__(self, dim: int, seq: int, device: torch.device = torch.device("cpu:0")):
        super().__init__()
        pe = torch.zeros(seq, dim)
        position = np.array([
            [0 if i == 0 else (pos / (10000 ** (2 * i / dim))) for i in range(dim)]
            for pos in range(seq)
        ], dtype=float)
        position[1:, 0::2] = np.sin(position[1:, 0::2])
        position[1:, 1::2] = np.cos(position[1:, 1::2])
        self.pos = torch.from_numpy(position).float().to(device)

    def forward(self, x: torch.Tensor):
        return x + self.pos


class Transformer(nn.Module):

    @staticmethod
    def get_attention_pad_mask(seq: torch.Tensor, seq_length: int, empty: int) -> torch.Tensor:
        """
        Get sequence mask
        :param seq: [N, seq_length]
        :param seq_length: seq_length
        :param empty: empty token
        :return:
        """
        msk = torch.eq(seq, empty).byte().unsqueeze(1)
        return msk.expand(seq.shape[0], seq_length, seq_length)

    @staticmethod
    def get_attention_sequence_mask(batch_size: int, seq_length: int, dev: torch.device = torch.device("cpu:0")):
        shape = (batch_size, seq_length, seq_length)
        subsequence_mask = torch.from_numpy(np.triu(np.ones(shape, dtype=int), k=1)).byte().to(dev)
        return subsequence_mask

    encoder: TransformerEncoder
    decoder: TransformerDecoder

    emb_encode: nn.Embedding
    emb_decode: nn.Embedding

    position_encode: PositionalEncoding
    position_decode: PositionalEncoding

    projection: nn.Module

    seq: int

    empty: int

    device: torch.device

    def __init__(self,
                 dict_size_encode: int, dict_size_decode: int,
                 d_emb: int, seq: int,
                 layer: int = 6,
                 head: int = 1, d_k: int = 1024, d_v: int = 1024,
                 attention_dropout: float = 0.1,
                 forward_hidden: int = 2048,
                 forward_dropout: float = 0.1,
                 emb_encode: torch.Tensor = None,
                 emb_decode: torch.Tensor = None,
                 empty: int = 0,
                 device: torch.device = torch.device("cpu:0")
                 ):
        super().__init__()

        self.seq = seq

        self.encoder = TransformerEncoder(layer, d_emb, head, d_k, d_v, attention_dropout, forward_hidden,
                                          forward_dropout)
        self.decoder = TransformerDecoder(layer, d_emb, head, d_k, d_v, attention_dropout, forward_hidden,
                                          forward_dropout)

        self.projection = nn.Sequential(
            nn.Linear(d_emb, dict_size_decode, bias=False)
        )

        self.position_encode = PositionalEncoding(d_emb, seq, device=device)
        self.position_decode = PositionalEncoding(d_emb, seq, device=device)

        self.emb_encode = nn.Embedding(dict_size_encode, d_emb) if emb_encode is None \
            else nn.Embedding.from_pretrained(emb_encode, freeze=True)

        self.emb_decode = nn.Embedding(dict_size_decode, d_emb) if emb_decode is None \
            else nn.Embedding.from_pretrained(emb_decode, freeze=True)

        self.empty = empty

        self.device = device

    def encode(self, encoder_input: torch.Tensor) -> torch.Tensor:
        mask = self.get_attention_pad_mask(encoder_input, self.seq, self.empty) == 1
        return self.encoder(self.position_encode(self.emb_encode(encoder_input)), mask)

    def decode(self, decoder_input: torch.Tensor, encoder_input: torch.Tensor, encoder_output: torch.Tensor):
        dec_mask = torch.gt(
            self.get_attention_sequence_mask(decoder_input.shape[0], self.seq, self.device)
            + self.get_attention_pad_mask(decoder_input, self.seq, self.empty),
            0
        )
        enc_mask = self.get_attention_pad_mask(encoder_input, self.seq, self.empty) == 1
        decoded = self.decoder(
            self.position_decode(self.emb_decode(decoder_input)),
            encoder_output, enc_mask, dec_mask
        )
        return self.projection(decoded)

    def forward(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor):
        enc = self.encode(encoder_input)
        return self.decode(decoder_input, encoder_input, enc)

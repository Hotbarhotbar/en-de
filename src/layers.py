import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None, relative_logits=None):
        # Q,K,V: (batch, heads, seq_len, d_k)
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (b,h,seq,seq)
        if relative_logits is not None:
            scores = scores + relative_logits  # add relative positional logits if available
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        return out, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, reduce_heads=False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads if not reduce_heads else max(1, n_heads//2)
        self.d_k = d_model // self.n_heads
        self.d_v = self.d_k
        self.W_Q = nn.Linear(d_model, self.n_heads * self.d_k)
        self.W_K = nn.Linear(d_model, self.n_heads * self.d_k)
        self.W_V = nn.Linear(d_model, self.n_heads * self.d_v)
        self.W_O = nn.Linear(self.n_heads * self.d_v, d_model)
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None, relative_logits=None):
        b, seq_q, _ = Q.size()
        seq_k = K.size(1)
        # linear projections
        q = self.W_Q(Q).view(b, seq_q, self.n_heads, self.d_k).transpose(1,2)  # (b,h,seq_q,d_k)
        k = self.W_K(K).view(b, seq_k, self.n_heads, self.d_k).transpose(1,2)
        v = self.W_V(V).view(b, seq_k, self.n_heads, self.d_v).transpose(1,2)
        out, attn = self.attention(q, k, v, mask=mask, relative_logits=relative_logits)
        out = out.transpose(1,2).contiguous().view(b, seq_q, -1)
        out = self.W_O(out)
        return out, attn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
    def forward(self, x):
        return self.net(x)

def subsequent_mask(size):
    # returns (1, size, size)
    attn_shape = (1, size, size)
    subsequent = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent == 0  # True where allowed
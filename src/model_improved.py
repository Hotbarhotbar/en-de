import torch
import torch.nn as nn
from layers import MultiHeadAttention, FeedForward, PositionalEncoding, subsequent_mask

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, ablation=None):
        super().__init__()
        self.ablation = ablation or {}
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, 
                                          reduce_heads=self.ablation.get("reduce_heads", False))
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # 预层归一化 (Pre-LN)
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, mask=src_mask)
        
        # --- 修复 3：实现 no_residual ---
        if self.ablation.get("no_residual", False):
            x = self.dropout(attn_out)  # 没有残差
        else:
            x = x + self.dropout(attn_out) # 有残差
        # ---------------------------------
        
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        
        # --- 修复 3：实现 no_residual ---
        if self.ablation.get("no_residual", False):
            x = self.dropout(ffn_out) # 没有残差
        else:
            x = x + self.dropout(ffn_out) # 有残差
        # ---------------------------------
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, ablation=None):
        super().__init__()
        self.ablation = ablation or {}
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, 
                                          reduce_heads=self.ablation.get("reduce_heads", False))
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout, 
                                           reduce_heads=self.ablation.get("reduce_heads", False))
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        # 预层归一化
        x_norm = self.norm1(x)
        self_out, _ = self.self_attn(x_norm, x_norm, x_norm, mask=tgt_mask)
        
        # --- 修复 3：实现 no_residual ---
        if self.ablation.get("no_residual", False):
            x = self.dropout(self_out)
        else:
            x = x + self.dropout(self_out)
        # ---------------------------------
        
        x_norm = self.norm2(x)
        cross_out, _ = self.cross_attn(x_norm, enc_out, enc_out, mask=src_mask)
        
        # --- 修复 3：实现 no_residual ---
        if self.ablation.get("no_residual", False):
            x = self.dropout(cross_out)
        else:
            x = x + self.dropout(cross_out)
        # ---------------------------------
        
        x_norm = self.norm3(x)
        ffn_out = self.ffn(x_norm)
        
        # --- 修复 3：实现 no_residual ---
        if self.ablation.get("no_residual", False):
            x = self.dropout(ffn_out)
        else:
            x = x + self.dropout(ffn_out)
        # ---------------------------------
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model=512, n_heads=8, d_ff=2048, 
                 num_layers=6, max_len=512, dropout=0.1, ablation=None, use_pos=True):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.trg_embed = nn.Embedding(trg_vocab, d_model)
        self.use_pos = use_pos
        
        # 只有在 use_pos 为 True 时才初始化
        self.pos_enc = PositionalEncoding(d_model, max_len) if self.use_pos else None
        
        self.ablation = ablation or {}
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, self.ablation) 
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, self.ablation) 
            for _ in range(num_layers)
        ])
        
        self.out = nn.Linear(d_model, trg_vocab)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src_ids, src_mask=None):
        x = self.src_embed(src_ids) * (self.d_model ** 0.5)
        if self.use_pos and self.pos_enc: # 检查 self.pos_enc 是否存在
            x = self.pos_enc(x)
        x = self.dropout(x)
        
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, trg_ids, enc_out, src_mask=None, tgt_mask=None):
        y = self.trg_embed(trg_ids) * (self.d_model ** 0.5)
        if self.use_pos and self.pos_enc: # 检查 self.pos_enc 是否存在
            y = self.pos_enc(y)
        y = self.dropout(y)
        
        for layer in self.decoder_layers:
            y = layer(y, enc_out, src_mask, tgt_mask)
        return y

    def forward(self, src_ids, trg_ids, src_mask=None, tgt_mask=None):
        enc = self.encode(src_ids, src_mask)
        dec = self.decode(trg_ids, enc, src_mask, tgt_mask)
        logits = self.out(dec)
        return logits
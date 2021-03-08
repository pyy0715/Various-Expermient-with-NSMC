import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0)
    pad_attn_mask = pad_attn_mask.unsqueeze(1).expand(batch_size, len_q, len_k) # pad_attn_mask: [batch_size, len_q, len_k]
    return pad_attn_mask

def scaled_dot_product_attention(query, key, value, mask=None):
    r'''Scaled Dot Product Attention
    Args:
        query: [batch_size, n_heads, len_query, d_model/n_heads]
        key: [batch_size, n_heads, len_key, d_model/n_heads]
        value: [batch_size, n_heads, len_value, d_model/n_heads]
        padding_mask : [batch_size, n_heads, len_query, len_key]
    '''
    d_k = key.shape[-1]
    # scores: [batch size, n_heads, len_quey, len_key]
    scores = torch.matmul(query, torch.transpose(key, -1, -2)) / np.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(scores, value) # output: [batch_size, n_heads, len_quey, d_model/n_heads]
    return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads

        self.W_Q = nn.Linear(self.d_model, self.d_model)
        self.W_K = nn.Linear(self.d_model, self.d_model)
        self.W_V = nn.Linear(self.d_model, self.d_model)

        self.W_0 = nn.Linear(self.d_model, self.d_model)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, Q, K, V, attn_mask):
        r'''
        Args:
            Q: [batch_size, len_q, d_model]
            K: [batch_size, len_k, d_model]
            V: [batch_size, len_v, d_model]
            attn_mask: [batch_size, len_q, len_k]
        '''

        residual, batch_size = Q, Q.size(0)

        # Split Heads
        # q_s: [batch_size, n_heads, len_q, d_model/num_heads]
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads,
                               self.head_dim).transpose(1, 2)
        # k_s: [batch_size, n_heads, len_k, d_model/num_heads]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads,
                               self.head_dim).transpose(1, 2)
        # v_s: [batch_size, n_heads, len_v, d_model/num_heads]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads,
                               self.head_dim).transpose(1, 2)

        # attn_mask : [batch_size, n_heads, len_q, len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        # output: [batch_size, n_heads, len_q, d_model/n_heads]
        output, attn = scaled_dot_product_attention(q_s, k_s, v_s, attn_mask)

        # Concatenate
        output = output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)  # output: [batch_size, len_q, d_model]

        output = self.W_0(output)
        output = self.layer_norm(output + residual)
        return output, attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super(PoswiseFeedForwardNet, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs  # inputs : [batch_size, len_q, d_model]
        output = self.linear1(inputs)
        output = self.dropout(F.relu(output))
        output = self.linear2(output)
        output = self.layer_norm(output + residual)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, dim_feedforward, dropout)

    def forward(self, enc_inputs, enc_mask):
        enc_outputs, attn_weight = self.attn(
            enc_inputs, enc_inputs, enc_inputs, enc_mask)  # enc_src to same Q,K,V
        # enc_outputs: [batch_size x len_q x d_model]
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn_weight


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, n_layers, d_model, n_heads, dim_feedforward, dropout):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([EncoderLayer(
            d_model, n_heads, dim_feedforward, dropout) for _ in range(n_layers)])
        self.scale = d_model

    def forward(self, enc_inputs): # enc_inputs: [batch_size, src_len] 
        enc_outputs = self.src_emb(enc_inputs)*math.sqrt(self.scale)
        enc_outputs = self.pos_emb(enc_outputs)
        enc_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        attn_weights = []
        for layer in self.layers:
            enc_outputs, attn_weight = layer(enc_outputs, enc_mask)
            attn_weights.append(attn_weight)
        return enc_outputs, attn_weights


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, n_heads)
        self.dec_enc_attn = MultiHeadAttention(d_model, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, dim_feedforward, dropout)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(
            dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(
            dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, n_layers, d_model, n_heads, dim_feedforward, dropout):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, dim_feedforward, dropout) for _ in range(n_layers)])
        
    def _generate_square_subsequent_mask(self, inputs, device='cpu'):
        mask = (torch.triu(torch.ones(inputs.shape[1], inputs.shape[1])) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.unsqueeze(0).repeat(inputs.shape[0], 1, 1)
        return mask.to(device)

    # dec_inputs : [batch_size x target_len]
    def forward(self, dec_inputs, enc_inputs, enc_outputs): 
        dec_outputs = self.tgt_emb(dec_inputs) 
        dec_outputs = self.pos_emb(dec_outputs)
        
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequent_mask = self._generate_square_subsequent_mask(dec_inputs)
        
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
        
        dec_self_attns = []
        dec_enc_attns = []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(
                dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns
    
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, 
                 n_layers: int = 6, d_model: int = 512, n_heads: int = 8, 
                 dim_feedforward: int = 2048, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, n_layers, d_model, n_heads, dim_feedforward, dropout)
        self.decoder = Decoder(tgt_vocab_size, n_layers, d_model, n_heads, dim_feedforward, dropout)
        self.linear = nn.Linear(d_model, tgt_vocab_size)
        self._reset_parameters()
        
    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        
    def forward(self, src_inputs, tgt_inputs):
        enc_outputs, enc_self_attns = self.encoder(src_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(
            tgt_inputs, src_inputs, enc_outputs)
        dec_logits = self.linear(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns

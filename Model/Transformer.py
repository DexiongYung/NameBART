import torch
import math
from torch import nn
from Constants import *


class Transformer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, d_model: int = 200, num_head: int = 8, num_e_layer: int = 6,
                 num_d_layer: int = 6, ff_dim: int = 1024, drop_out: float = 0.1):
        '''
        Args:
            input_dim: Size of the vocab of the input
            output_dim: Size of the vocab for output
            num_head: Number of heads in mutliheaded attention models
            num_e_layer: Number of sub-encoder layers
            num_d_layer: Number of sub-decoder layers
            ff_dim: Dimension of feedforward network in mulihead models
            d_model: The dimension to embed input and output features into
            drop_out: The drop out percentage
        '''
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.transformer = nn.Transformer(d_model, num_head, num_e_layer, num_d_layer, ff_dim, drop_out, activation='gelu')
        self.decoder_embedder = nn.Embedding(output_dim, d_model)
        self.encoder_embedder = nn.Embedding(input_dim, d_model)
        self.fc1 = nn.Linear(d_model, output_dim)
        self.softmax = nn.Softmax(dim=2)
        self.positional_encoder = PositionalEncoding(d_model, drop_out)
        self.to(DEVICE)

    def forward(self, src: torch.Tensor, trg: torch.Tensor, src_mask: torch.Tensor = None,
                trg_mask: torch.Tensor = None):
        embedded_src = self.positional_encoder(self.encoder_embedder(src) * math.sqrt(self.d_model))
        embedded_trg = self.positional_encoder(self.decoder_embedder(trg) * math.sqrt(self.d_model))
        output = self.transformer.forward(embedded_src, embedded_trg, src_mask, trg_mask)
        return self.softmax(self.fc1(output))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

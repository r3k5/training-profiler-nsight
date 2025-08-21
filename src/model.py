import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyTransformerLM(nn.Module):
    """
    A very small Transformer language model:
    - Token + positional embeddings
    - TransformerEncoder with N layers
    - Linear head to vocab
    Works on synthetic integer tokens.
    """
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (B, S, D)
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.tok_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lm_head.bias)

    def forward(self, tokens: torch.Tensor):
        """
        tokens: LongTensor of shape (B, S)
        returns: logits (B, S, vocab)
        """
        B, S = tokens.shape
        pos = torch.arange(0, S, device=tokens.device, dtype=torch.long)
        pos = pos.unsqueeze(0).expand(B, S)

        x = self.tok_embed(tokens) + self.pos_embed(pos)
        x = self.encoder(x)  # (B, S, D)
        x = self.ln(x)
        logits = self.lm_head(x)  # (B, S, V)
        return logits

    def loss(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Cross-entropy loss; shift targets by one if you want next-token LM.
        Here we simply predict the same positions for simplicity.
        """
        B, S, V = logits.shape
        logits = logits.view(B * S, V)
        targets = targets.view(B * S)
        return F.cross_entropy(logits, targets)
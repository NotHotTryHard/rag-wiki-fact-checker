import torch
import torch.nn as nn
from transformers import AutoModel


class FactChecker(nn.Module):
    def __init__(self, model_name, num_labels=3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.head = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        """
        input_ids:      (batch, top_k, seq_len)
        attention_mask:  (batch, top_k, seq_len)
        returns:         (batch,) truthfulness scores
        """
        B, K, L = input_ids.shape
        # flatten to (B*K, L) for encoder
        out = self.encoder(
            input_ids=input_ids.view(B * K, L),
            attention_mask=attention_mask.view(B * K, L),
        )
        cls = out.last_hidden_state[:, 0]       # CLS token state (B*K, H)
        logits = self.head(cls).view(B, K, -1)  # (B, K, num_labels)

        # max-pool over K evidences
        pooled = logits.max(dim=1).values        # (B, num_labels)

        return pooled
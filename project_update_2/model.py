"""Dual-head causal model for streaming code-switch prediction.

One shared multilingual encoder (mBERT or XLM-R) reads the prefix x_1..x_t.
A causal mask stops position t from attending to any later position.
Two linear heads read the hidden state at position t:
  switch head   -> 2 scores: does x_{t+1} change language? (0 = no, 1 = yes)
  duration head -> 3 scores: length bin of the new burst (0 = 1-2, 1 = 3-6, 2 = 7+ subwords)
"""
from torch import nn
from transformers import AutoModel

# Short names -> Hugging Face model IDs. Any full model ID also works.
BACKBONES = {
    'mbert': 'google-bert/bert-base-multilingual-cased',
    'xlmr': 'FacebookAI/xlm-roberta-base',
}


class DualHeadModel(nn.Module):
    def __init__(self, backbone='mbert', revision='main', dropout=0.1, causal=True):
        super().__init__()
        name = BACKBONES.get(backbone, backbone)
        # is_decoder=True: the encoder builds a lower-triangular attention mask,
        # so position t can only attend to positions 0..t (no look-ahead).
        # causal=False exists only for the negative control in the causality test.
        # add_pooling_layer=False: we use every token's hidden state, not the [CLS] pooler.
        self.backbone = AutoModel.from_pretrained(
            name, revision=revision, is_decoder=causal, add_pooling_layer=False)
        hidden = self.backbone.config.hidden_size          # 768 for both base models
        self.dropout = nn.Dropout(dropout)
        self.switch_head = nn.Linear(hidden, 2)            # no switch / switch
        self.duration_head = nn.Linear(hidden, 3)          # Small / Medium / Large

    def forward(self, input_ids, attention_mask):
        """input_ids, attention_mask: (B, T) -> switch logits (B, T, 2), duration logits (B, T, 3)."""
        # use_cache=False: decoder mode would otherwise keep key/value caches we never use.
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        h = self.dropout(out.last_hidden_state)            # (B, T, hidden): one vector per position
        # Both heads read the same h, so both losses send gradients into the shared backbone.
        return self.switch_head(h), self.duration_head(h)

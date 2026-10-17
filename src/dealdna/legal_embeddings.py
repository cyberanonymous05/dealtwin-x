"""Embed legal clause passages (MAC clauses, termination fee, earn-out language) extracted from merger filings using FinBERT."""

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

_MODEL_NAME = "yiyanghkust/finbert-tone"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModel.from_pretrained(_MODEL_NAME).to(_DEVICE)
        _model.eval()
    return _tokenizer, _model


def legal_embedding_dim() -> int:
    _, model = _load_model()
    return model.config.hidden_size


@torch.no_grad()
def embed_legal_clauses(passages: list[str]) -> np.ndarray:
    """Mean-pool FinBERT token embeddings per passage, then average across a deal's clause passages."""
    tokenizer, model = _load_model()
    if not passages:
        return np.zeros(model.config.hidden_size, dtype=float)

    inputs = tokenizer(passages, padding=True, truncation=True, max_length=128, return_tensors="pt").to(_DEVICE)
    outputs = model(**inputs)
    hidden = outputs.last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    doc_embedding = pooled.mean(dim=0)
    return doc_embedding.cpu().numpy().astype(float)

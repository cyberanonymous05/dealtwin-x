"""Stage 4 -- text embeddings (GPU).

  legal_finbert.npy   FinBERT-tone mean-pooled over each selected clause passage, averaged per deal
  legal_legalbert.npy LEGAL-BERT, same passages
  legal_mpnet.npy     all-mpnet-base-v2 sentence embeddings, same passages
  business_mpnet.npy  all-mpnet-base-v2 over the target's 10-K business description (<=16 chunks of
                      ~200 words, averaged); zero vector when no 10-K was available
Rows are aligned with data/processed/deal_universe.json (deal_ids.json records the order).
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging_utils import get_logger

logger = get_logger("build_embeddings")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join("data", "embeddings")
HF_ENCODERS = {"finbert": "yiyanghkust/finbert-tone", "legalbert": "nlpaueb/legal-bert-base-uncased"}
MPNET = "sentence-transformers/all-mpnet-base-v2"


@torch.no_grad()
def hf_mean_pool(model_name: str, docs: list[list[str]]) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE).eval().half()
    flat = [(i, p) for i, ps in enumerate(docs) for p in ps]
    sums = np.zeros((len(docs), model.config.hidden_size))
    counts = np.zeros(len(docs))
    for start in range(0, len(flat), 256):
        batch = flat[start: start + 256]
        enc = tok([p for _, p in batch], padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
        hidden = model(**enc).last_hidden_state.float()
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = ((hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)).cpu().numpy()
        for (i, _), vec in zip(batch, pooled):
            sums[i] += vec
            counts[i] += 1
    return sums / np.maximum(counts, 1)[:, None]


def st_mean(model, docs: list[list[str]]) -> np.ndarray:
    flat = [(i, p) for i, ps in enumerate(docs) for p in ps]
    emb = model.encode([p for _, p in flat], batch_size=128, show_progress_bar=False, normalize_embeddings=True) if flat else []
    out = np.zeros((len(docs), model.get_sentence_embedding_dimension()))
    counts = np.zeros(len(docs))
    for (i, _), vec in zip(flat, emb):
        out[i] += vec
        counts[i] += 1
    return out / np.maximum(counts, 1)[:, None]


def chunks(text: str, words: int = 200, max_chunks: int = 16) -> list[str]:
    toks = text.split()
    return [" ".join(toks[k: k + words]) for k in range(0, min(len(toks), words * max_chunks), words)]


def main():
    from sentence_transformers import SentenceTransformer

    with open(os.path.join("data", "processed", "deal_universe.json")) as f:
        deals = json.load(f)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "deal_ids.json"), "w") as f:
        json.dump([d["deal_id"] for d in deals], f)

    passages = [d.get("clause_passages") or [] for d in deals]
    for name, model_name in HF_ENCODERS.items():
        np.save(os.path.join(OUT, f"legal_{name}.npy"), hf_mean_pool(model_name, passages).astype(np.float32))
        logger.info(f"legal_{name} done")

    mpnet = SentenceTransformer(MPNET, device=DEVICE)
    np.save(os.path.join(OUT, "legal_mpnet.npy"), st_mean(mpnet, passages).astype(np.float32))
    business = [chunks(d.get("business_description") or "") for d in deals]
    np.save(os.path.join(OUT, "business_mpnet.npy"), st_mean(mpnet, business).astype(np.float32))
    logger.info(f"business_mpnet done ({sum(1 for b in business if b)} deals with a 10-K description)")


if __name__ == "__main__":
    main()

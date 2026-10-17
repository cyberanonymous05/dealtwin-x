"""Text-similarity baselines for precedent retrieval, built from the SAME clause passages DealDNA's
FinBERT legal block consumes. These answer the reviewer question the naive industry+size baseline
cannot: does a domain-adapted contextual embedding (FinBERT) of the selected legal passages actually
beat a plain lexical model (TF-IDF, BM25) or a generic sentence encoder (MiniLM) on those passages?

Each function returns an (n x n) similarity matrix aligned to `deals` order, usable anywhere the
DealDNA / naive-baseline similarity matrices are (ablation study, outcome-grounded study, significance).
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_MINILM_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_minilm = None


def _deal_documents(deals: list[dict]) -> list[str]:
    """One text document per deal: its selected legal-clause passages concatenated (same passages the
    FinBERT legal block embeds). Falls back to an empty string when a deal has no scorable passages."""
    docs = []
    for d in deals:
        passages = d.get("clause_passages") or []
        docs.append(" ".join(p.strip() for p in passages) if passages else "")
    return docs


def _cosine_gram(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normed = matrix / np.where(norms == 0, 1.0, norms)
    return normed @ normed.T


def tfidf_similarity(deals: list[dict]) -> np.ndarray:
    docs = _deal_documents(deals)
    vec = TfidfVectorizer(stop_words="english", min_df=2, max_df=0.9, ngram_range=(1, 2), sublinear_tf=True)
    tfidf = vec.fit_transform(docs).toarray()
    return _cosine_gram(tfidf).astype(float)


def bm25_similarity(deals: list[dict], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25) -> np.ndarray:
    """Okapi BM25 with the same formula and defaults as rank_bm25.BM25Okapi, vectorised with sparse
    matrices: score(q, d) = sum over q's tokens t of idf(t) * tf(t,d)(k1+1) / (tf(t,d) + k1(1-b+b|d|/avgdl))."""
    from sklearn.feature_extraction.text import CountVectorizer

    docs = [doc.lower() if doc else "__empty__" for doc in _deal_documents(deals)]
    counts = CountVectorizer(token_pattern=r"\S+", lowercase=False).fit_transform(docs).tocsr().astype(float)
    n_docs = counts.shape[0]
    doc_len = np.asarray(counts.sum(axis=1)).ravel()
    df = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log(n_docs - df + 0.5) - np.log(df + 0.5)
    floor = epsilon * idf.mean()
    idf = np.where(idf < 0, floor, idf)  # rank_bm25 replaces negative idf by epsilon * mean idf
    norm = k1 * (1 - b + b * doc_len / doc_len.mean())
    w = counts.copy()
    rows = np.repeat(np.arange(n_docs), np.diff(w.indptr))
    w.data = w.data * (k1 + 1) / (w.data + norm[rows])
    w = w.multiply(idf[None, :]).tocsr()
    sim = (counts @ w.T).toarray()
    # BM25 scores are asymmetric and unbounded; symmetrize and row-normalize to [0, 1] for comparability.
    sim = (sim + sim.T) / 2.0
    row_max = sim.max(axis=1, keepdims=True)
    return sim / np.where(row_max == 0, 1.0, row_max)


def minilm_similarity(deals: list[dict]) -> np.ndarray:
    global _minilm
    if _minilm is None:
        from sentence_transformers import SentenceTransformer

        _minilm = SentenceTransformer(_MINILM_NAME)
    docs = _deal_documents(deals)
    emb = _minilm.encode(
        [doc if doc else "no legal clause passages available" for doc in docs],
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return (np.asarray(emb) @ np.asarray(emb).T).astype(float)


TEXT_BASELINES = {
    "tfidf_legal_text": tfidf_similarity,
    "bm25_legal_text": bm25_similarity,
    "minilm_legal_text": minilm_similarity,
}

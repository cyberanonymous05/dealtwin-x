"""Semantic (embedding-based) legal clause selection: classify filing sentences into clause categories
by cosine similarity to prototype descriptions, instead of literal regex keyword matching.

This is the NLP step in DealDNA's legal-clause pipeline. clause_extraction.py's regex is reserved for
genuinely numeric deal-term parsing (cash-per-share, termination-fee dollar amounts) -- inherently
pattern-based, not a clause-understanding task. This module is what lets the FinBERT legal embedding
represent clauses phrased without the literal keyword (e.g. "Company shall not encourage a competing
offer" without the word "solicitation"), which a keyword-window regex would miss entirely.
"""

import re

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None
_prototype_centroids = None

CLAUSE_PROTOTYPES = {
    "material_adverse_effect": [
        "No event, change, or circumstance has occurred that constitutes a Material Adverse Effect on "
        "the company's business, financial condition, or results of operations.",
        "The parties disagree about whether a material adverse change permits termination of the merger agreement.",
    ],
    "termination_fee": [
        "The company must pay the acquirer a termination fee if the merger agreement is terminated "
        "under specified circumstances.",
        "A reverse termination fee is payable by the parent if regulatory approval is not obtained.",
    ],
    "earnout_contingent_value": [
        "Stockholders will receive contingent value rights or earn-out payments based on future milestones.",
        "Additional consideration is payable upon achievement of specified post-closing performance targets.",
    ],
    "no_solicitation_fiduciary_out": [
        "The company shall not solicit, initiate, or encourage competing acquisition proposals from third parties.",
        "The board may change its recommendation in response to a superior proposal under the fiduciary out provision.",
    ],
    "representations_warranties": [
        "The company makes representations and warranties regarding its financial statements, litigation, "
        "and compliance with law.",
        "Each party represents and warrants that it has the requisite corporate authority to enter into the agreement.",
    ],
    "regulatory_approval": [
        "Consummation of the merger is subject to receipt of required regulatory approvals, including antitrust clearance.",
        "The parties must make filings under the Hart-Scott-Rodino Act and obtain governmental consents.",
    ],
    "appraisal_rights": [
        "Stockholders who do not vote in favor of the merger may be entitled to exercise appraisal rights under state law.",
        "Dissenting shareholders may seek judicial appraisal of the fair value of their shares.",
    ],
    "exchange_ratio_consideration": [
        "Each share of common stock will be converted into the right to receive a fixed exchange ratio of acquirer stock.",
        "Merger consideration consists of a combination of cash and shares of the acquiring company.",
    ],
}

# Keyword anchors per clause type. They are NOT used to choose passages for the encoders by default;
# they define (1) the keyword-anchored silver standard used to measure selector precision per clause type
# and (2) the optional lexical gate applied to the clause types whose semantic precision is lowest.
KEYWORD_ANCHORS = {
    "material_adverse_effect": re.compile(r"material\s+adverse\s+(?:effect|change)", re.I),
    "termination_fee": re.compile(r"termination\s+fee|break[\s-]?up\s+fee", re.I),
    "earnout_contingent_value": re.compile(r"earn[\s-]?out|contingent\s+value|\bCVRs?\b|milestone", re.I),
    "no_solicitation_fiduciary_out": re.compile(r"solicit|superior\s+proposal|acquisition\s+proposal|fiduciary|change\s+(?:of|in)\s+recommendation", re.I),
    "representations_warranties": re.compile(r"represent\w*\s+and\s+warrant|representations", re.I),
    "regulatory_approval": re.compile(r"antitrust|Hart[\s-]Scott|\bHSR\b|regulatory\s+approv|governmental\s+(?:approval|consent)|competition\s+law|CFIUS", re.I),
    "appraisal_rights": re.compile(r"appraisal|dissent", re.I),
    "exchange_ratio_consideration": re.compile(r"exchange\s+ratio|merger\s+consideration|right\s+to\s+receive|per\s+share", re.I),
}


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;])\s+(?=[A-Z(])")
_KEYWORD_FILTER_RE = re.compile(
    r"\b(shall|agreement|company|parent|merger|stockholders?|shares?|fee|effect|material|"
    r"terminat|represent|warrant|regulator|approval|solicit|appraisal|consideration|closing)\b",
    re.IGNORECASE,
)


def _load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def _load_prototype_centroids() -> dict[str, np.ndarray]:
    global _prototype_centroids
    if _prototype_centroids is None:
        model = _load_model()
        centroids = {}
        for clause_type, prototypes in CLAUSE_PROTOTYPES.items():
            embeddings = model.encode(prototypes, normalize_embeddings=True)
            centroid = embeddings.mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-9
            centroids[clause_type] = centroid
        _prototype_centroids = centroids
    return _prototype_centroids


def _segment_candidate_sentences(text: str, min_len: int = 60, max_len: int = 500, max_candidates: int = 1400) -> list[str]:
    candidates = []
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        sentence = sentence.strip()
        if min_len <= len(sentence) <= max_len and _KEYWORD_FILTER_RE.search(sentence):
            candidates.append(sentence)
        if len(candidates) >= max_candidates:
            break
    return candidates


def encode_sentences(sentences: list[str]) -> np.ndarray:
    return _load_model().encode(sentences, batch_size=256, show_progress_bar=False, normalize_embeddings=True)


def select_from_embeddings(
    candidates: list[str], candidate_embeddings: np.ndarray, top_n_per_type: int = 2,
    similarity_threshold: float = 0.35, lexical_gate: tuple = (),
) -> list[dict]:
    """Pick up to `top_n_per_type` passages per clause type by cosine similarity to the type centroid,
    deduplicated across types. Types listed in `lexical_gate` additionally require their keyword anchor.
    Returns [{"type", "text", "score"}]."""
    centroids = _load_prototype_centroids()
    selected, seen = [], set()
    if not candidates:
        return selected
    for clause_type, centroid in centroids.items():
        scores = candidate_embeddings @ centroid
        picked = 0
        for idx in np.argsort(-scores):
            if scores[idx] < similarity_threshold:
                break
            sentence = candidates[idx]
            if sentence in seen:
                continue
            if clause_type in lexical_gate and not KEYWORD_ANCHORS[clause_type].search(sentence):
                continue
            seen.add(sentence)
            selected.append({"type": clause_type, "text": sentence, "score": float(scores[idx])})
            picked += 1
            if picked >= top_n_per_type:
                break
    return selected


def segment_candidate_sentences(text: str) -> list[str]:
    return _segment_candidate_sentences(text)


def classify_clause_passages(text: str, top_n_per_type: int = 2, similarity_threshold: float = 0.35) -> list[str]:
    """Backward-compatible wrapper: passages only."""
    candidates = _segment_candidate_sentences(text)
    if not candidates:
        return []
    picked = select_from_embeddings(candidates, encode_sentences(candidates), top_n_per_type, similarity_threshold)
    return [p["text"] for p in picked]

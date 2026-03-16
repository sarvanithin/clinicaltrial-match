"""
Semantic similarity search using the in-memory EmbeddingIndex.

Computes patient query vector and searches the trial index.
"""

from __future__ import annotations

from clinicaltrial_match.infrastructure.embeddings import EmbeddingIndex
from clinicaltrial_match.patients.models import PatientFeatures


class SemanticSearcher:
    def __init__(self, embeddings: EmbeddingIndex) -> None:
        self._embeddings = embeddings

    def search(
        self,
        features: PatientFeatures,
        top_k: int = 30,
    ) -> list[tuple[str, float]]:
        """Return list of (nct_id, cosine_similarity) sorted descending."""
        query_text = features.clinical_summary or _build_fallback_query(features)
        if not query_text.strip():
            return []
        query_vec = self._embeddings.encode_one(query_text)
        return self._embeddings.search(query_vec, top_k)


def _build_fallback_query(features: PatientFeatures) -> str:
    parts = []
    if features.age_years:
        parts.append(f"{int(features.age_years)} year old {features.gender.value}")
    if features.diagnoses:
        parts.append("with " + ", ".join(d.name for d in features.diagnoses[:5]))
    if features.medications:
        active = [m.name for m in features.medications if m.active]
        if active:
            parts.append("taking " + ", ".join(active[:3]))
    return " ".join(parts) or "patient"

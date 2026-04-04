"""
Semantic similarity search using the in-memory EmbeddingIndex.

Falls back to keyword search when the embedding model is not yet loaded,
so matching works immediately without waiting for model warm-up.
"""

from __future__ import annotations

from clinicaltrial_match.infrastructure.embeddings import EmbeddingIndex
from clinicaltrial_match.patients.models import PatientFeatures

# Common medication → implied condition keywords (helps when FHIR only has meds, no diagnoses)
_MED_TO_CONDITION: dict[str, list[str]] = {
    "metformin": ["diabetes", "type 2 diabetes"],
    "insulin": ["diabetes", "type 1 diabetes"],
    "glipizide": ["diabetes", "type 2 diabetes"],
    "glimepiride": ["diabetes", "type 2 diabetes"],
    "sitagliptin": ["diabetes", "type 2 diabetes"],
    "empagliflozin": ["diabetes", "heart failure"],
    "lisinopril": ["hypertension", "heart failure"],
    "atorvastatin": ["cardiovascular", "hyperlipidemia"],
    "amlodipine": ["hypertension"],
    "losartan": ["hypertension"],
    "tamoxifen": ["breast cancer"],
    "letrozole": ["breast cancer"],
    "trastuzumab": ["breast cancer"],
    "levodopa": ["parkinson"],
    "donepezil": ["alzheimer", "dementia"],
    "memantine": ["alzheimer", "dementia"],
}


class SemanticSearcher:
    def __init__(self, embeddings: EmbeddingIndex) -> None:
        self._embeddings = embeddings

    def search(
        self,
        features: PatientFeatures,
        top_k: int = 30,
    ) -> list[tuple[str, float]]:
        """Synchronous search — use search_async in async contexts."""
        if not self._embeddings.has_index:
            return []
        query_text = features.clinical_summary or _build_fallback_query(features)
        if not query_text.strip():
            return []
        query_vec = self._embeddings.encode_one(query_text)
        return self._embeddings.search(query_vec, top_k)

    async def search_async(
        self,
        features: PatientFeatures,
        top_k: int = 30,
        db=None,
    ) -> list[tuple[str, float]]:
        """Async-safe search.

        Uses semantic (embedding) search when the model is ready; falls back
        to keyword search against the DB so results are always returned.
        """
        if not self._embeddings.has_index:
            return []

        # Fast keyword fallback when model hasn't loaded yet
        if not self._embeddings.is_model_ready:
            return _keyword_search(features, db, top_k)

        query_text = features.clinical_summary or _build_fallback_query(features)
        if not query_text.strip():
            return _keyword_search(features, db, top_k)

        query_vec = await self._embeddings.encode_one_async(query_text)
        results = self._embeddings.search(query_vec, top_k)

        # If semantic search returns nothing useful, blend with keyword results
        if not results or results[0][1] < 0.2:
            kw = _keyword_search(features, db, top_k)
            seen = {r[0] for r in results}
            for nct_id, score in kw:
                if nct_id not in seen:
                    results.append((nct_id, score))
            results = results[:top_k]

        return results


def _extract_keywords(features: PatientFeatures) -> list[str]:
    """Build a deduplicated keyword list from patient diagnoses + medications."""
    keywords: list[str] = []
    for d in features.diagnoses:
        if d.name:
            keywords.append(d.name)
    for m in features.medications:
        if m.name:
            keywords.append(m.name)
            # Expand medication to implied conditions
            for implied in _MED_TO_CONDITION.get(m.name.lower(), []):
                keywords.append(implied)
    # Also pull words from clinical summary (skip stop words)
    stop = {"with", "on", "and", "the", "a", "an", "patient", "year", "old", "unknown"}
    for word in (features.clinical_summary or "").split():
        w = word.strip(".,;:-").lower()
        if len(w) > 3 and w not in stop:
            keywords.append(w)
    # Deduplicate preserving order
    seen: set[str] = set()
    out: list[str] = []
    for k in keywords:
        if k.lower() not in seen:
            seen.add(k.lower())
            out.append(k)
    return out


def _keyword_search(features: PatientFeatures, db, top_k: int) -> list[tuple[str, float]]:
    """Keyword-based fallback — no model required."""
    if db is None:
        return []
    keywords = _extract_keywords(features)
    if not keywords:
        return []
    rows = db.keyword_search_trials(keywords, limit=top_k)
    # Normalise score to 0–1 range (max possible = len(keywords))
    max_kw = max(len(keywords), 1)
    results: list[tuple[str, float]] = []
    for row in rows:
        text = " ".join(str(row.get(f, "") or "") for f in ("title", "conditions", "brief_summary", "eligibility_text")).lower()
        hits = sum(1 for kw in keywords if kw.lower() in text)
        score = min(hits / max_kw, 1.0)
        results.append((row["nct_id"], score))
    return results


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

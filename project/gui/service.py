"""
Service layer for the Hospital Recommendation System GUI.

Wraps the existing pipeline (Gemini subject extraction + SBERT embedding ranking)
into a single clean function that the Flask app can call.
"""

import pandas as pd
import torch

from project.utils import config
from project.model.loader import get_model
from project.model.embeddings import load_or_create_embed
from project.model.ranking_hospital import get_query_embedding, extract_hospital_rank_k
from project.gemini.subject_extraction import extract_subject

# ---------------------------------------------------------------------------
# Singleton resources – loaded once at import time
# ---------------------------------------------------------------------------

_hospital_info = None
_hospital_embed = None


def _ensure_loaded():
    """Lazily load CSV + embeddings the first time they are needed."""
    global _hospital_info, _hospital_embed
    if _hospital_info is None:
        hospital_info_path = config.DATA_DIR / "hospital_info.csv"
        _hospital_info = pd.read_csv(hospital_info_path)

        embed_path = config.CACHE_DIR / "hospital_embeddings(sroberta-sts-normal,address0.2).pt"
        model = get_model("jhgan/ko-sroberta-sts")
        _hospital_embed = load_or_create_embed(_hospital_info, model, embed_path)


def recommend_hospitals(symptom_text: str, top_k: int = 10):
    """
    End-to-end pipeline: symptom text → ranked hospital list.

    Returns
    -------
    dict with keys:
        extracted_dept : str   – department extracted by Gemini
        hospitals      : list[dict] – top-k hospitals with detail fields
    """
    _ensure_loaded()

    # 1. Extract medical department via Gemini
    extracted_dept = extract_subject(symptom_text)

    # 2. Encode query and rank hospitals
    query_embed = get_query_embedding(symptom_text)
    candidate_df = extract_hospital_rank_k(top_k, _hospital_info, query_embed, _hospital_embed)

    # 3. Convert DataFrame → list of dicts for JSON serialization
    display_columns = [
        "hospital_name",
        "telephone",
        "opening_hours",
        "medical_subject",
        "address",
    ]
    hospitals = []
    for _, row in candidate_df.iterrows():
        hospitals.append({col: str(row.get(col, "")) for col in display_columns})

    return {
        "extracted_dept": extracted_dept or "",
        "hospitals": hospitals,
    }

# Hospital Recommendation System: Project Pipeline Analysis

This document provides a detailed technical overview of the AI pipeline used in the Hospital Recommendation System.

## 1. Project Overview
The system is designed to provide personalized hospital recommendations based on user-described symptoms. It leverages a hybrid approach combining **Large Language Models (LLM)** for semantic understanding and **Sentence Embeddings** for efficient retrieval.

---

## 2. Technical Architecture

### Phase A: Semantic Subject Extraction (Gemini)
When a user enters symptoms, the system utilizes the **Google Gemini 2.0 Flash** model to bridge the gap between "layman's terms" and "medical terminology."

1.  **Initial Analysis:** The raw symptom text is sent to Gemini to understand the medical context.
2.  **Subject Refinement:** A second targeted prompt is used to extract the specific **Expected Medical Department** (e.g., "내과", "안과").
3.  **Prompt Engineering:** The system explicitly requests "plain text" without markdown formatting or special characters to ensure the output is compatible with the downstream embedding model.
4.  **Localization:** The term `"성북구"` (Seongbuk-gu) is appended to the extracted department to prioritize local results during the similarity search.

### Phase B: Advanced Hospital Embedding Strategy
Unlike simple text search, this system uses a **Weighted Multi-Field Embedding** approach to represent hospitals in a high-dimensional vector space.

*   **Base Model:** `jhgan/ko-sroberta-sts`
    *   *Type:* SBERT (Sentence-BERT) based on the RoBERTa architecture.
    *   *Specialization:* Fine-tuned specifically for Korean Semantic Textual Similarity (STS).
*   **Weighted Combination:** 
    For every hospital in the database (`hospital_info.csv`), a single representative vector is created by embedding four fields separately and combining them using specific weights:
    | Field | Weight | Description |
    | :--- | :--- | :--- |
    | `medical_subject` | **0.4** | The primary factor for matching symptoms. |
    | `address` | **0.4** | Ensures geographic relevance. |
    | `hospital_name` | **0.1** | Provides identity context. |
    | `opening_hours` | **0.1** | Adds temporal context. |
*   **Caching:** To ensure low-latency responses, these embeddings are pre-computed and stored as `.pt` (PyTorch) files in the `cache/` directory.

### Phase C: Semantic Ranking & Retrieval
The retrieval process finds the "mathematical nearest neighbors" to the user's query.

1.  **Query Vectorization:** The query (Extracted Subject + "성북구") is converted into a vector using the same `ko-sroberta-sts` model.
2.  **Similarity Calculation:** The system computes the **Cosine Similarity** between the query vector and the pre-computed hospital embedding matrix.
3.  **Top-K Retrieval:** The `k` hospitals (default: 10) with the highest similarity scores are selected.
4.  **Formatting:** Results are mapped back to the original CSV data and displayed to the user with details including Name, Subject, and Address.

---

## 3. Key Specifications

| Component | Specification |
| :--- | :--- |
| **LLM Model** | `gemini-2.0-flash` |
| **Embedding Model** | `jhgan/ko-sroberta-sts` (HuggingFace) |
| **Embedding Dimension** | 768 dimensions |
| **Similarity Metric** | Cosine Similarity |
| **Data Format** | CSV (Pandas) |
| **Vector Storage** | PyTorch Serialization (`.pt`) |

---

## 4. Pipeline Flow Summary
`User Symptom` -> `Gemini (Subject Extraction)` -> `SBERT (Query Encoding)` -> `Cosine Similarity vs. Weighted Hospital Matrix` -> `Top-10 Ranking` -> `UI Display`

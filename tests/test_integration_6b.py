"""Autograder tests for Integration 6B — NER + Embeddings Semantic Pipeline."""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from semantic_pipeline import (
    load_and_preprocess, run_ner, compute_embeddings,
    semantic_search, enrich_with_entities, demonstrate_pipeline,
)


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "climate_articles.csv")
QUERIES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "example_queries.txt")

# Sample texts for lightweight tests
SAMPLE_TEXTS = [
    "The United Nations climate summit in Paris established carbon reduction targets.",
    "Rising sea levels threaten coastal cities in Southeast Asia and the Pacific Islands.",
    "Renewable energy investments in Europe have grown significantly since 2020.",
    "Deforestation in the Amazon contributes to global carbon dioxide emissions.",
    "Water scarcity affects millions of people in the Middle East and North Africa.",
]


# ── Load and Preprocess ──────────────────────────────────────────────────

def test_load_and_preprocess():
    """load_and_preprocess should return a DataFrame with a 'text' column."""
    df = load_and_preprocess(DATA_PATH)
    assert df is not None, "load_and_preprocess returned None"
    assert isinstance(df, pd.DataFrame), "Must return a pandas DataFrame"
    assert "text" in df.columns, "DataFrame must have a 'text' column"
    assert len(df) > 0, "DataFrame should not be empty"
    # No null texts
    assert df["text"].notna().all(), "Text column should have no null values"


# ── NER ──────────────────────────────────────────────────────────────────

def test_run_ner():
    """run_ner should extract entities and return a structured DataFrame."""
    entities = run_ner(SAMPLE_TEXTS)
    assert entities is not None, "run_ner returned None"
    assert isinstance(entities, pd.DataFrame), "Must return a pandas DataFrame"
    for col in ["text_index", "entity_text", "entity_label"]:
        assert col in entities.columns, f"Missing required column: {col}"
    # Sample texts contain clear entities (United Nations, Paris, etc.)
    assert len(entities) > 0, "Should extract at least some entities from sample texts"


# ── Embeddings ───────────────────────────────────────────────────────────

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


@pytest.mark.skipif(not _HAS_TRANSFORMERS, reason="transformers not installed")
def test_compute_embeddings():
    """compute_embeddings should return an (n, 768) array."""
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()
    embs = compute_embeddings(SAMPLE_TEXTS, tokenizer, model)
    assert embs is not None, "compute_embeddings returned None"
    assert isinstance(embs, np.ndarray), "Must return a numpy array"
    assert embs.shape == (len(SAMPLE_TEXTS), 768), (
        f"Expected shape ({len(SAMPLE_TEXTS)}, 768), got {embs.shape}"
    )
    assert not np.allclose(embs, 0), "Embeddings should not be all zeros"


# ── Semantic Search ──────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_TRANSFORMERS, reason="transformers not installed")
def test_semantic_search():
    """semantic_search should return top-k results sorted by similarity."""
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()
    embs = compute_embeddings(SAMPLE_TEXTS, tokenizer, model)
    assert embs is not None

    # Use the first text's embedding as query
    query_emb = embs[0]
    results = semantic_search(query_emb, embs, SAMPLE_TEXTS, top_k=3)
    assert results is not None, "semantic_search returned None"
    assert isinstance(results, list), "Must return a list"
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    for item in results:
        assert isinstance(item, tuple) and len(item) == 2, (
            f"Each result must be a (text, score) tuple, got {item}"
        )
        text, score = item
        assert isinstance(text, str), "Result text must be a string"
        assert -1.0 <= score <= 1.01, f"Similarity score out of range: {score}"
    # Results should be sorted by score descending
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True), "Results must be sorted by similarity descending"


# ── Enrich with Entities ─────────────────────────────────────────────────

def test_enrich_with_entities():
    """enrich_with_entities should attach entity info to search results."""
    # Build mock search results and entity DataFrame
    search_results = [
        (SAMPLE_TEXTS[0], 0.95),
        (SAMPLE_TEXTS[1], 0.80),
    ]
    entity_df = pd.DataFrame({
        "text_index": [0, 0, 1],
        "entity_text": ["United Nations", "Paris", "Southeast Asia"],
        "entity_label": ["ORG", "GPE", "LOC"],
    })
    enriched = enrich_with_entities(search_results, entity_df)
    assert enriched is not None, "enrich_with_entities returned None"
    assert isinstance(enriched, list), "Must return a list"
    assert len(enriched) == 2, f"Expected 2 enriched results, got {len(enriched)}"
    for item in enriched:
        assert isinstance(item, dict), "Each enriched result must be a dict"
        assert "text" in item, "Missing 'text' key"
        assert "similarity" in item, "Missing 'similarity' key"
        assert "entities" in item, "Missing 'entities' key"
        assert isinstance(item["entities"], list), "'entities' must be a list"


# ── Full Pipeline Demo ───────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_TRANSFORMERS, reason="transformers not installed")
def test_demonstrate_pipeline():
    """demonstrate_pipeline should return results for all queries."""
    # Use a small subset for speed
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()

    corpus_df = pd.DataFrame({"text": SAMPLE_TEXTS})
    entities = run_ner(SAMPLE_TEXTS)
    assert entities is not None
    embs = compute_embeddings(SAMPLE_TEXTS, tokenizer, model)
    assert embs is not None

    queries = ["climate agreements on carbon reduction"]
    results = demonstrate_pipeline(corpus_df, entities, embs, queries)
    assert results is not None, "demonstrate_pipeline returned None"
    assert isinstance(results, dict), "Must return a dict"
    assert len(results) >= 1, "Should have results for at least one query"
    for q, enriched in results.items():
        assert isinstance(enriched, list), "Query results must be a list"
        assert len(enriched) > 0, "Should return at least one result per query"

# Integration 6B — NER + Embeddings Semantic Pipeline

Build an end-to-end NLP pipeline combining named entity recognition and embedding-based semantic search on a climate article corpus.

## Objectives

- Load and preprocess a text corpus for NLP processing
- Extract named entities using spaCy
- Compute DistilBERT embeddings for semantic search
- Implement cosine-similarity-based semantic search
- Enrich search results with extracted entities
- Demonstrate and analyze the full pipeline

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Data

- `data/climate_articles.csv` — Climate article corpus
- `data/example_queries.txt` — Three example search queries for pipeline demonstration

## Tasks

Complete all six functions in `semantic_pipeline.py`:

1. **`load_and_preprocess(filepath)`** — Load the dataset and prepare texts
2. **`run_ner(texts)`** — Extract named entities using spaCy
3. **`compute_embeddings(texts, tokenizer, model)`** — Compute DistilBERT embeddings for all texts
4. **`semantic_search(query, corpus_embeddings, corpus_texts, top_k=5)`** — Find the most similar texts
5. **`enrich_with_entities(search_results, entity_df)`** — Add NER entities to search results
6. **`demonstrate_pipeline(corpus_df, entity_df, embeddings, queries)`** — Run the full pipeline on example queries

## Submission

1. Create a branch named `integration-6b-semantic-pipeline`
2. Complete all functions in `semantic_pipeline.py`
3. Run `pytest tests/ -v` to verify your work
4. Open a PR to `main` — the autograder will run automatically
5. Your PR description must include:
   - Pipeline demo output for 3 queries
   - Semantic vs. keyword search comparison
   - Production improvement proposals
   - Paste your PR URL into TalentLMS → Module 6 → Integration 6B to submit this assignment

Resubmissions are accepted through Saturday of the assignment week.

---

## License

This repository is provided for educational use only. See [LICENSE](LICENSE) for terms.

You may clone and modify this repository for personal learning and practice, and reference code you wrote here in your professional portfolio. Redistribution outside this course is not permitted.

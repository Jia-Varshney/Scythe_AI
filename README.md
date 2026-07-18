# Scythe AI: An Automated Multi-Agent Pipeline for Curation of Agricultural Machine Learning Literature

Builds a SQLite dataset of agriculture-related research papers using Semantic
Scholar discovery, LangChain-assisted filtering, and Scite DOI enrichment.

Architecture diagram: [docs/architecture.md](docs/architecture.md)

## Setup

```bash
python3 -m pip install -r requirements.txt
export GROQ_API_KEY="..."
export SEMANTIC_SCHOLAR_API_KEY="..." # optional, improves rate limits
export SCITE_API_KEY="..."            # optional for public tallies/papers endpoints
```

## Run

```bash
python3 pipeline.py --sqlite-path sciteai_papers.sqlite3
```

Use deterministic keyword filtering without LangChain/OpenAI:

```bash
python3 pipeline.py --no-llm --max-results-per-query 25
```

Add custom search terms:

```bash
python3 pipeline.py --query '"grape disease detection" PyTorch'
```

Run only a specific query without the built-in defaults:

```bash
python3 pipeline.py --no-default-queries --query '"leaf disease detection" deep learning agriculture'
```

Resume a specific query from a later Semantic Scholar offset:

```bash
python3 pipeline.py --no-default-queries --query '"crop yield forecasting" PyTorch' --semantic-scholar-start-offset 25
```

The SQLite output includes normalized `papers`, `authors`, `paper_authors`,
`search_runs`, and raw `api_enrichment` payloads.

## LLM Refinement

After crawling, refine saved SQLite papers without calling Semantic Scholar or
Scite again:

```bash
export GROQ_API_KEY="..."
python3 pipeline.py refine-with-llm --sqlite-path sciteai_papers.sqlite3 --refine-limit 25 --dry-run
```

Use OpenAI instead of Groq:

```bash
export OPENAI_API_KEY="..."
python3 pipeline.py refine-with-llm --llm-provider openai --sqlite-path sciteai_papers.sqlite3 --refine-limit 25 --dry-run
```

Remove `--dry-run` to update `papers` in place and store each analysis payload
as `llm_refinement` in `api_enrichment`:

```bash
python3 pipeline.py refine-with-llm --sqlite-path sciteai_papers.sqlite3 --refine-limit 25
```

By default, `refine-with-llm` skips papers that already have an
`llm_refinement` payload. Run it repeatedly in batches to process the remaining
papers:

```bash
python3 pipeline.py refine-with-llm --sqlite-path sciteai_papers.sqlite3 --refine-limit 100
```

To intentionally reprocess already-refined papers:

```bash
python3 pipeline.py refine-with-llm --include-already-refined --refine-limit 25 --dry-run
```

## Vector Search

Build a local SQLite vector index from the SQLite papers:

```bash
python3 pipeline.py build-vector-index --sqlite-path sciteai_papers.sqlite3
```

Search the index:

```bash
python3 pipeline.py search-vector "PyTorch crop disease detection using leaf images" --top-k 10
```

Include the Scite paper link in the printed results and export the same rows to CSV:

```bash
python3 pipeline.py search-vector "PyTorch crop disease detection using leaf images" --top-k 10 --export-csv results.csv
```

If a query returns too few useful hits, have the pipeline fetch more papers from
Semantic Scholar for that same prompt, add them to the vector store, and then
refine the newly inserted papers with Groq:

```bash
python3 pipeline.py search-or-expand "insect patterns in the United States Midwest"
```

That command also supports CSV export:

```bash
python3 pipeline.py search-or-expand "insect patterns in the United States Midwest" --export-csv locust_results.csv
```

The default vector backend is `sqlite`, which stores the paper documents and
metadata in the same SQLite database and scores queries with TF-IDF over those
stored documents. `--vector-index-path` and `--vector-collection` only matter if
you switch to `--vector-backend chroma`. The default local embedding label is
`tfidf-bigrams`. Use `--min-vector-hits` and `--min-vector-score` to decide when
the pipeline should auto-expand the corpus from Semantic Scholar, and
`--prompt-variants` to control how many related query variants it generates first.
The crawl step itself stays heuristic; Groq is only used after new papers are
inserted to refine their metadata.
The search output now includes a Scite paper link when one is available from the
stored `scite_paper` enrichment payload.

# SciteAI Architecture

```mermaid
flowchart LR
    U[User / CLI] --> C{Command}

    C -->|crawl| S2[Semantic Scholar Search]
    C -->|refine-with-llm| LLM[Groq / OpenAI Refinement]
    C -->|build-vector-index| VIDX[Build Local Vector Index]
    C -->|search-vector| VSRCH[Search Local Vector Index]
    C -->|search-or-expand| SOE[Search or Expand]

    subgraph Discovery["Discovery and Ingestion"]
        S2 --> QA[Heuristic Query Relevance]
        QA --> PA[Heuristic Paper Analysis]
        PA --> SC[Scite DOI Enrichment]
        SC --> DB[(SQLite)]
        PA --> DB
        QA --> DB
    end

    subgraph Refinement["LLM Refinement"]
        DB --> LLM
        LLM --> DB
    end

    subgraph Indexing["Vector Indexing"]
        DB --> VIDX
        VIDX --> VT[(vector_documents in SQLite)]
    end

    subgraph Search["Search and Expansion"]
        VSRCH --> VT
        SOE --> PV[Prompt Variant Expansion]
        PV --> VT
        SOE -->|thin results| S2
        SOE -->|new papers| DB
        SOE -->|new papers| LLM
    end

    DB --> CSV[CSV Export]
    VSRCH --> CSV
    SOE --> CSV

    S2 --- S2API[Semantic Scholar API]
    SC --- SciteAPI[Scite API]
    LLM --- GroqAPI[Groq API]
    LLM --- OpenAIAPI[OpenAI API]
```

## Notes

- SQLite is the source of truth for papers, authors, raw API payloads, and run history.
- `vector_documents` is a local search index backed by SQLite.
- `search-or-expand` uses heuristic prompt variants first, then crawls Semantic Scholar only if local coverage is thin.
- Groq/OpenAI is only used for post-ingest refinement, not for discovery.
- Scite enriches DOI-backed records with citation tallies and paper metadata when available.

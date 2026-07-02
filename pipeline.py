#!/usr/bin/env python3
"""LangChain-assisted literature pipeline for agriculture AI papers.

The pipeline searches Semantic Scholar, classifies agriculture methodology papers,
enriches DOI-backed records with Scite tallies/metadata, and writes normalized
records plus raw API payloads into SQLite with a vector index and exports results to csv file.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import logging
import os
import random
import re
import sqlite3
import time
import csv
from typing import Any, Iterable
from urllib.parse import quote

import requests

try:
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
except ImportError:  # LangChain is optional at import time; see PaperAnalyzer.
    ChatOpenAI = None
    ChatPromptTemplate = None
    JsonOutputParser = None

try:
    import chromadb
    from chromadb.utils import embedding_functions
except Exception:
    chromadb = None
    embedding_functions = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    TfidfVectorizer = None


LOGGER = logging.getLogger("sciteai.pipeline")
DEFAULT_VECTOR_EMBEDDING_MODEL = "tfidf-bigrams"


SEMANTIC_SCHOLAR_FIELDS = ",".join(
    [
        "paperId",
        "externalIds",
        "url",
        "title",
        "abstract",
        "venue",
        "year",
        "authors",
        "citationCount",
        "influentialCitationCount",
        "fieldsOfStudy",
        "s2FieldsOfStudy",
        "publicationTypes",
        "openAccessPdf",
        "tldr",
    ]
)


DEFAULT_QUERIES = [
    '"crop disease detection" "computer vision"',
    '"plant disease detection" PyTorch',
    '"leaf disease detection" deep learning agriculture',
    '"crop yield prediction" "time series"',
    '"crop yield forecasting" PyTorch',
    '"agricultural yield forecasting" JAX',
    '"remote sensing" "crop yield" "time series"',
]


AGRICULTURE_TERMS = {
    "agriculture",
    "agricultural",
    "crop",
    "crops",
    "plant",
    "plants",
    "leaf",
    "leaves",
    "wheat",
    "rice",
    "maize",
    "corn",
    "soybean",
    "tomato",
    "potato",
    "cassava",
    "vineyard",
    "orchard",
    "locust",
    "locusts",
    "pest",
    "pests",
    "insect",
    "insects",
    "outbreak",
    "outbreaks",
    "swarm",
    "swarms",
}

CV_DISEASE_TERMS = {
    "disease detection",
    "plant disease",
    "crop disease",
    "leaf disease",
    "computer vision",
    "image classification",
    "object detection",
    "segmentation",
    "convolutional",
    "cnn",
    "vision transformer",
}

YIELD_FORECAST_TERMS = {
    "yield prediction",
    "yield forecasting",
    "crop yield",
    "time series",
    "temporal",
    "lstm",
    "gru",
    "transformer",
    "forecasting",
    "remote sensing",
}

FRAMEWORK_TERMS = {
    "pytorch": "PyTorch",
    "torch": "PyTorch",
    "jax": "JAX",
    "tensorflow": "TensorFlow",
    "keras": "TensorFlow/Keras",
}


@dataclasses.dataclass(frozen=True)
class Config:
    semantic_scholar_api_key: str | None
    scite_api_key: str | None
    sqlite_path: str
    max_results_per_query: int
    page_size: int
    semantic_scholar_start_offset: int
    min_relevance_score: float
    request_timeout: float
    max_retries: int
    base_backoff_seconds: float
    semantic_scholar_base_url: str
    scite_base_url: str
    semantic_scholar_delay_seconds: float
    llm_provider: str
    llm_model: str
    llm_delay_seconds: float
    use_llm: bool

    @classmethod
    def from_env(cls, args: argparse.Namespace) -> "Config":
        return cls(
            semantic_scholar_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
            scite_api_key=os.getenv("SCITE_API_KEY"),
            sqlite_path=args.sqlite_path,
            max_results_per_query=args.max_results_per_query,
            page_size=args.page_size,
            semantic_scholar_start_offset=args.semantic_scholar_start_offset,
            min_relevance_score=args.min_relevance_score,
            request_timeout=args.request_timeout,
            max_retries=args.max_retries,
            base_backoff_seconds=args.base_backoff_seconds,
            semantic_scholar_base_url=args.semantic_scholar_base_url.rstrip("/"),
            scite_base_url=args.scite_base_url.rstrip("/"),
            semantic_scholar_delay_seconds=args.semantic_scholar_delay_seconds,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model or default_llm_model(args.llm_provider),
            llm_delay_seconds=args.llm_delay_seconds,
            use_llm=not args.no_llm,
        )


class RateLimitedClient:
    def __init__(self, timeout: float, max_retries: int, base_backoff_seconds: float) -> None:
        self.session = requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_backoff_seconds = base_backoff_seconds

    def request_json(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any] | list[Any]:
        last_error: Exception | None = None
        last_status: int | None = None
        last_body = ""
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    timeout=self.timeout,
                )
                last_status = response.status_code
                last_body = response.text[:500]
                if response.status_code == 404:
                    return {}
                if response.status_code in {429, 500, 502, 503, 504}:
                    self._sleep_before_retry(response, attempt)
                    continue
                response.raise_for_status()
                if not response.content:
                    return {}
                return response.json()
            except (requests.RequestException, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                self._sleep_before_retry(None, attempt)
        detail = f"last_status={last_status}"
        if last_body:
            detail = f"{detail} last_body={last_body!r}"
        raise RuntimeError(f"Request failed after retries: {method} {url} ({detail})") from last_error

    def _sleep_before_retry(self, response: requests.Response | None, attempt: int) -> None:
        retry_after = response.headers.get("Retry-After") if response is not None else None
        if retry_after:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = self.base_backoff_seconds
        else:
            delay = self.base_backoff_seconds * (2**attempt) + random.uniform(0, 0.25)
        time.sleep(delay)


class SemanticScholarClient:
    def __init__(self, config: Config, http: RateLimitedClient) -> None:
        self.config = config
        self.http = http

    def search(self, query: str) -> Iterable[dict[str, Any]]:
        headers = {}
        if self.config.semantic_scholar_api_key:
            headers["x-api-key"] = self.config.semantic_scholar_api_key

        offset = self.config.semantic_scholar_start_offset
        yielded = 0
        while yielded < self.config.max_results_per_query:
            limit = min(self.config.page_size, self.config.max_results_per_query - yielded)
            payload = self.http.request_json(
                "GET",
                f"{self.config.semantic_scholar_base_url}/paper/search",
                headers=headers,
                params={
                    "query": query,
                    "offset": offset,
                    "limit": limit,
                    "fields": SEMANTIC_SCHOLAR_FIELDS,
                },
            )
            if not isinstance(payload, dict):
                return
            papers = payload.get("data") or []
            if not papers:
                return
            for paper in papers:
                yielded += 1
                yield paper
            offset += len(papers)
            total = payload.get("total")
            if total is not None and offset >= int(total):
                return
            if self.config.semantic_scholar_delay_seconds > 0:
                time.sleep(self.config.semantic_scholar_delay_seconds)


class SciteClient:
    def __init__(self, config: Config, http: RateLimitedClient) -> None:
        self.config = config
        self.http = http

    def get_tallies(self, doi: str) -> dict[str, Any]:
        return self._get(f"/tallies/{quote(doi, safe='')}")

    def get_paper(self, doi: str) -> dict[str, Any]:
        return self._get(f"/papers/{quote(doi, safe='')}")

    def _get(self, path: str) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        if self.config.scite_api_key:
            headers["Authorization"] = f"Bearer {self.config.scite_api_key}"
        payload = self.http.request_json("GET", f"{self.config.scite_base_url}{path}", headers=headers)
        return payload if isinstance(payload, dict) else {}


class PaperAnalyzer:
    def __init__(self, config: Config) -> None:
        self.chain = None
        if langchain_is_available(config):
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You classify research papers for a dataset. Return only valid JSON with keys: "
                        "is_agriculture_related, is_cv_disease_detection, is_yield_forecasting, "
                        "task_type, framework, code_url, dataset_names, relevance_score, rationale. "
                        "Scores are 0 to 1. Prefer PyTorch or JAX implementation details.",
                    ),
                    (
                        "human",
                        "Title: {title}\nAbstract: {abstract}\nVenue: {venue}\nYear: {year}\n"
                        "Fields: {fields}\nOpen access: {open_access}\nTLDR: {tldr}",
                    ),
                ]
            )
            self.chain = prompt | build_chat_model(config) | JsonOutputParser()

    def analyze(self, paper: dict[str, Any]) -> dict[str, Any]:
        if self.chain:
            try:
                result = self.chain.invoke(self._prompt_payload(paper))
                return self._normalize_result(result, paper)
            except Exception as exc:  # Keep long crawls moving if one LLM call fails.
                LOGGER.warning("LLM analysis failed for %s: %s", paper.get("paperId"), exc)
        return self._heuristic_analysis(paper)

    def _prompt_payload(self, paper: dict[str, Any]) -> dict[str, Any]:
        external = paper.get("externalIds") or {}
        return {
            "title": paper.get("title") or "",
            "abstract": paper.get("abstract") or "",
            "venue": paper.get("venue") or "",
            "year": paper.get("year") or "",
            "fields": paper.get("fieldsOfStudy") or paper.get("s2FieldsOfStudy") or [],
            "open_access": paper.get("openAccessPdf") or {},
            "tldr": paper.get("tldr") or {},
            "doi": external.get("DOI") or external.get("doi") or "",
        }

    def _heuristic_analysis(self, paper: dict[str, Any]) -> dict[str, Any]:
        text = normalized_text(
            " ".join(
                [
                    str(paper.get("title") or ""),
                    str(paper.get("abstract") or ""),
                    json.dumps(paper.get("tldr") or {}),
                ]
            )
        )
        is_ag = any(term in text for term in AGRICULTURE_TERMS)
        is_cv = any(term in text for term in CV_DISEASE_TERMS)
        is_yield = any(term in text for term in YIELD_FORECAST_TERMS)
        framework = detect_framework(text)
        code_url = extract_code_url(text)
        task_type = infer_task_type(is_cv, is_yield)
        score = score_relevance(is_ag, is_cv, is_yield, framework, code_url)
        return {
            "is_agriculture_related": is_ag,
            "is_cv_disease_detection": is_cv,
            "is_yield_forecasting": is_yield,
            "task_type": task_type,
            "framework": framework,
            "code_url": code_url,
            "dataset_names": [],
            "relevance_score": score,
            "rationale": "keyword fallback classifier",
        }

    def _normalize_result(self, result: dict[str, Any], paper: dict[str, Any]) -> dict[str, Any]:
        fallback = self._heuristic_analysis(paper)
        normalized = dict(fallback)
        normalized.update({key: result.get(key, fallback.get(key)) for key in normalized})
        normalized["is_agriculture_related"] = parse_bool(normalized["is_agriculture_related"])
        normalized["is_cv_disease_detection"] = parse_bool(normalized["is_cv_disease_detection"])
        normalized["is_yield_forecasting"] = parse_bool(normalized["is_yield_forecasting"])
        normalized["relevance_score"] = clamp_float(normalized.get("relevance_score"), fallback["relevance_score"])
        normalized["task_type"] = scalar_text(normalized.get("task_type"))
        normalized["framework"] = scalar_text(normalized.get("framework"))
        normalized["code_url"] = scalar_text(normalized.get("code_url"))
        if isinstance(normalized.get("dataset_names"), str):
            normalized["dataset_names"] = [normalized["dataset_names"]]
        if not isinstance(normalized.get("dataset_names"), list):
            normalized["dataset_names"] = []
        return normalized


class SearchPlanner:
    def __init__(self, config: Config) -> None:
        self.chain = None
        if langchain_is_available(config):
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Generate focused Semantic Scholar keyword queries for agriculture AI literature. "
                        "Return only a JSON array of strings. Cover computer vision crop disease detection, "
                        "time-series crop-yield forecasting, and PyTorch/JAX implementations.",
                    ),
                    ("human", "Seed queries:\n{seed_queries}"),
                ]
            )
            self.chain = prompt | build_chat_model(config) | JsonOutputParser()

    def plan(self, seed_queries: list[str]) -> list[str]:
        if not self.chain:
            return list(dict.fromkeys(seed_queries))
        try:
            planned = self.chain.invoke({"seed_queries": "\n".join(seed_queries)})
            if isinstance(planned, list):
                queries = [str(query).strip() for query in planned if str(query).strip()]
                return list(dict.fromkeys(seed_queries + queries))
        except Exception as exc:
            LOGGER.warning("LLM query planning failed; using seed queries: %s", exc)
        return list(dict.fromkeys(seed_queries))


class PromptExpander:
    def __init__(self, config: Config) -> None:
        self.chain = None
        if langchain_is_available(config):
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Expand a user search prompt into a small set of related Semantic Scholar search queries. "
                        "Return only a JSON array of short keyword queries. Include the original intent plus close synonyms, "
                        "broader variants, and adjacent phrasing. Avoid redundant duplicates.",
                    ),
                    ("human", "Prompt: {prompt}"),
                ]
            )
            self.chain = prompt | build_chat_model(config) | JsonOutputParser()

    def expand(self, prompt: str, max_variants: int = 5) -> list[str]:
        if self.chain:
            try:
                expanded = self.chain.invoke({"prompt": prompt})
                if isinstance(expanded, list):
                    queries = [normalize_query_variant(str(query)) for query in expanded if normalize_query_variant(str(query))]
                    return limit_unique_queries([prompt] + queries, max_variants + 1)
            except Exception as exc:
                LOGGER.warning("Prompt expansion failed; using heuristic variants: %s", exc)
        return heuristic_prompt_variants(prompt, max_variants=max_variants)


class QueryRelevanceAnalyzer:
    def __init__(self, config: Config) -> None:
        self.chain = None
        if langchain_is_available(config):
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You judge whether a paper is relevant to a user search query. "
                        "Return only valid JSON with keys: is_relevant, relevance_score, task_type, framework, code_url, dataset_names, rationale. "
                        "The score is 0 to 1. Prefer a high score only when the paper is substantively about the query.",
                    ),
                    (
                        "human",
                        "Query: {query}\nTitle: {title}\nAbstract: {abstract}\nVenue: {venue}\nYear: {year}\nFields: {fields}",
                    ),
                ]
            )
            self.chain = prompt | build_chat_model(config) | JsonOutputParser()

    def analyze(self, query: str, paper: dict[str, Any]) -> dict[str, Any]:
        if self.chain:
            try:
                result = self.chain.invoke(self._prompt_payload(query, paper))
                return self._normalize_result(result, query, paper)
            except Exception as exc:
                LOGGER.warning("Query relevance analysis failed for %s: %s", paper.get("paperId"), exc)
        return self._heuristic_analysis(query, paper)

    def _prompt_payload(self, query: str, paper: dict[str, Any]) -> dict[str, Any]:
        return {
            "query": query,
            "title": paper.get("title") or "",
            "abstract": paper.get("abstract") or "",
            "venue": paper.get("venue") or "",
            "year": paper.get("year") or "",
            "fields": paper.get("fieldsOfStudy") or paper.get("s2FieldsOfStudy") or [],
        }

    def _heuristic_analysis(self, query: str, paper: dict[str, Any]) -> dict[str, Any]:
        query_tokens = set(tokenize(query))
        paper_tokens = set(tokenize(" ".join([str(paper.get("title") or ""), str(paper.get("abstract") or "")])))
        overlap = query_tokens & paper_tokens
        score = 0.15 * len(overlap)
        if overlap:
            score += 0.35
        if any(term in normalized_text(query) for term in AGRICULTURE_TERMS) or any(term in paper_tokens for term in AGRICULTURE_TERMS):
            score += 0.1
        score = min(score, 1.0)
        return {
            "is_relevant": score >= 0.2,
            "relevance_score": score,
            "task_type": "query_expansion",
            "framework": detect_framework(normalized_text(" ".join([str(paper.get("title") or ""), str(paper.get("abstract") or "")]))),
            "code_url": extract_code_url(normalized_text(" ".join([str(paper.get("title") or ""), str(paper.get("abstract") or "")]))),
            "dataset_names": [],
            "rationale": "token overlap fallback",
        }

    def _normalize_result(self, result: dict[str, Any], query: str, paper: dict[str, Any]) -> dict[str, Any]:
        fallback = self._heuristic_analysis(query, paper)
        normalized = dict(fallback)
        normalized.update({key: result.get(key, fallback.get(key)) for key in normalized})
        normalized["is_relevant"] = parse_bool(normalized["is_relevant"])
        normalized["relevance_score"] = clamp_float(normalized.get("relevance_score"), fallback["relevance_score"])
        normalized["task_type"] = scalar_text(normalized.get("task_type")) or "query_expansion"
        normalized["framework"] = scalar_text(normalized.get("framework"))
        normalized["code_url"] = scalar_text(normalized.get("code_url"))
        if isinstance(normalized.get("dataset_names"), str):
            normalized["dataset_names"] = [normalized["dataset_names"]]
        if not isinstance(normalized.get("dataset_names"), list):
            normalized["dataset_names"] = []
        return normalized


class SQLiteStore:
    def __init__(self, path: str) -> None:
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        self.init_sqlite_vector_schema(DEFAULT_VECTOR_EMBEDDING_MODEL)

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            PRAGMA journal_mode = WAL;

            CREATE TABLE IF NOT EXISTS papers (
                paper_id TEXT PRIMARY KEY,
                semantic_scholar_id TEXT,
                doi TEXT,
                title TEXT NOT NULL,
                abstract TEXT,
                year INTEGER,
                venue TEXT,
                url TEXT,
                citation_count INTEGER,
                influential_citation_count INTEGER,
                task_type TEXT,
                is_agriculture_related INTEGER NOT NULL,
                is_cv_disease_detection INTEGER NOT NULL,
                is_yield_forecasting INTEGER NOT NULL,
                framework TEXT,
                code_url TEXT,
                dataset_names TEXT,
                relevance_score REAL NOT NULL,
                scite_supporting_count INTEGER,
                scite_contradicting_count INTEGER,
                scite_mentioning_count INTEGER,
                scite_total_count INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS authors (
                author_id TEXT PRIMARY KEY,
                name TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS paper_authors (
                paper_id TEXT NOT NULL,
                author_id TEXT NOT NULL,
                author_order INTEGER NOT NULL,
                PRIMARY KEY (paper_id, author_id),
                FOREIGN KEY (paper_id) REFERENCES papers(paper_id),
                FOREIGN KEY (author_id) REFERENCES authors(author_id)
            );

            CREATE TABLE IF NOT EXISTS api_enrichment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT NOT NULL,
                source TEXT NOT NULL,
                raw_json TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
            );

            CREATE TABLE IF NOT EXISTS search_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                results_seen INTEGER NOT NULL DEFAULT 0,
                results_kept INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        self.conn.commit()

    def start_run(self, query: str) -> int:
        cursor = self.conn.execute(
            "INSERT INTO search_runs (query, started_at) VALUES (?, ?)",
            (query, utc_now()),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def complete_run(self, run_id: int, seen: int, kept: int) -> None:
        self.conn.execute(
            """
            UPDATE search_runs
            SET completed_at = ?, results_seen = ?, results_kept = ?
            WHERE id = ?
            """,
            (utc_now(), seen, kept, run_id),
        )
        self.conn.commit()

    def upsert_paper(
        self,
        paper: dict[str, Any],
        analysis: dict[str, Any],
        scite_tallies: dict[str, Any] | None,
        scite_paper: dict[str, Any] | None,
    ) -> str:
        external = paper.get("externalIds") or {}
        doi = external.get("DOI") or external.get("doi")
        paper_id = paper.get("paperId") or stable_id(doi or paper.get("title") or "")
        now = utc_now()
        tallies = scite_tallies or {}
        self.conn.execute(
            """
            INSERT INTO papers (
                paper_id, semantic_scholar_id, doi, title, abstract, year, venue, url,
                citation_count, influential_citation_count, task_type,
                is_agriculture_related, is_cv_disease_detection, is_yield_forecasting,
                framework, code_url, dataset_names, relevance_score,
                scite_supporting_count, scite_contradicting_count, scite_mentioning_count,
                scite_total_count, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
                doi = excluded.doi,
                title = excluded.title,
                abstract = excluded.abstract,
                year = excluded.year,
                venue = excluded.venue,
                url = excluded.url,
                citation_count = excluded.citation_count,
                influential_citation_count = excluded.influential_citation_count,
                task_type = excluded.task_type,
                is_agriculture_related = excluded.is_agriculture_related,
                is_cv_disease_detection = excluded.is_cv_disease_detection,
                is_yield_forecasting = excluded.is_yield_forecasting,
                framework = excluded.framework,
                code_url = excluded.code_url,
                dataset_names = excluded.dataset_names,
                relevance_score = excluded.relevance_score,
                scite_supporting_count = excluded.scite_supporting_count,
                scite_contradicting_count = excluded.scite_contradicting_count,
                scite_mentioning_count = excluded.scite_mentioning_count,
                scite_total_count = excluded.scite_total_count,
                updated_at = excluded.updated_at
            """,
            (
                paper_id,
                paper.get("paperId"),
                doi,
                paper.get("title") or "",
                paper.get("abstract"),
                paper.get("year"),
                paper.get("venue"),
                paper.get("url"),
                paper.get("citationCount"),
                paper.get("influentialCitationCount"),
                analysis.get("task_type"),
                int(bool(analysis.get("is_agriculture_related"))),
                int(bool(analysis.get("is_cv_disease_detection"))),
                int(bool(analysis.get("is_yield_forecasting"))),
                analysis.get("framework"),
                analysis.get("code_url"),
                json.dumps(analysis.get("dataset_names") or []),
                float(analysis.get("relevance_score") or 0),
                count_from_tallies(tallies, "supporting"),
                count_from_tallies(tallies, "contradicting"),
                count_from_tallies(tallies, "mentioning"),
                count_from_tallies(tallies, "total"),
                now,
                now,
            ),
        )
        self._upsert_authors(paper_id, paper.get("authors") or [])
        self.add_raw_payload(paper_id, "semantic_scholar", paper)
        if scite_tallies:
            self.add_raw_payload(paper_id, "scite_tallies", scite_tallies)
        if scite_paper:
            self.add_raw_payload(paper_id, "scite_paper", scite_paper)
        self.sync_vector_document(paper_id=paper_id, paper=paper, analysis=analysis, scite_tallies=scite_tallies)
        self.conn.commit()
        return paper_id

    def add_raw_payload(self, paper_id: str, source: str, payload: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO api_enrichment (paper_id, source, raw_json, fetched_at)
            VALUES (?, ?, ?, ?)
            """,
            (paper_id, source, json.dumps(payload, sort_keys=True), utc_now()),
        )

    def _upsert_authors(self, paper_id: str, authors: list[dict[str, Any]]) -> None:
        for order, author in enumerate(authors):
            author_id = author.get("authorId") or stable_id(author.get("name") or f"{paper_id}-{order}")
            name = author.get("name") or "Unknown"
            self.conn.execute(
                """
                INSERT INTO authors (author_id, name)
                VALUES (?, ?)
                ON CONFLICT(author_id) DO UPDATE SET name = excluded.name
                """,
                (author_id, name),
            )
            self.conn.execute(
                """
                INSERT INTO paper_authors (paper_id, author_id, author_order)
                VALUES (?, ?, ?)
                ON CONFLICT(paper_id, author_id) DO UPDATE SET author_order = excluded.author_order
                """,
                (paper_id, author_id, order),
            )

    def iter_papers_for_refinement(
        self,
        *,
        limit: int | None,
        min_current_relevance_score: float,
        include_already_refined: bool,
    ) -> Iterable[dict[str, Any]]:
        sql = """
            SELECT
                paper_id, semantic_scholar_id, doi, title, abstract, year, venue, url,
                citation_count, influential_citation_count, relevance_score
            FROM papers
            WHERE relevance_score >= ?
        """
        params: list[Any] = [min_current_relevance_score]
        if not include_already_refined:
            sql = f"""
                {sql}
                AND NOT EXISTS (
                    SELECT 1
                    FROM api_enrichment
                    WHERE api_enrichment.paper_id = papers.paper_id
                      AND api_enrichment.source = 'llm_refinement'
                )
            """
        sql = f"{sql} ORDER BY relevance_score DESC, citation_count DESC"
        if limit is not None:
            sql = f"{sql} LIMIT ?"
            params.append(limit)
        for row in self.conn.execute(sql, params):
            yield {
                "paperId": row["semantic_scholar_id"] or row["paper_id"],
                "externalIds": {"DOI": row["doi"]} if row["doi"] else {},
                "title": row["title"],
                "abstract": row["abstract"],
                "year": row["year"],
                "venue": row["venue"],
                "url": row["url"],
                "citationCount": row["citation_count"],
                "influentialCitationCount": row["influential_citation_count"],
                "_db_paper_id": row["paper_id"],
            }

    def iter_papers_by_ids(self, paper_ids: list[str]) -> Iterable[dict[str, Any]]:
        if not paper_ids:
            return []
        placeholders = ",".join("?" for _ in paper_ids)
        sql = f"""
            SELECT
                paper_id, semantic_scholar_id, doi, title, abstract, year, venue, url,
                citation_count, influential_citation_count, relevance_score
            FROM papers
            WHERE paper_id IN ({placeholders})
            ORDER BY relevance_score DESC, citation_count DESC
        """
        for row in self.conn.execute(sql, paper_ids):
            yield {
                "paperId": row["semantic_scholar_id"] or row["paper_id"],
                "externalIds": {"DOI": row["doi"]} if row["doi"] else {},
                "title": row["title"],
                "abstract": row["abstract"],
                "year": row["year"],
                "venue": row["venue"],
                "url": row["url"],
                "citationCount": row["citation_count"],
                "influentialCitationCount": row["influential_citation_count"],
                "_db_paper_id": row["paper_id"],
            }

    def update_paper_analysis(self, paper_id: str, analysis: dict[str, Any]) -> None:
        normalized = normalize_analysis_for_storage(analysis)
        self.conn.execute(
            """
            UPDATE papers
            SET
                task_type = ?,
                is_agriculture_related = ?,
                is_cv_disease_detection = ?,
                is_yield_forecasting = ?,
                framework = ?,
                code_url = ?,
                dataset_names = ?,
                relevance_score = ?,
                updated_at = ?
            WHERE paper_id = ?
            """,
            (
                normalized["task_type"],
                int(normalized["is_agriculture_related"]),
                int(normalized["is_cv_disease_detection"]),
                int(normalized["is_yield_forecasting"]),
                normalized["framework"],
                normalized["code_url"],
                json.dumps(normalized["dataset_names"]),
                normalized["relevance_score"],
                utc_now(),
                paper_id,
            ),
        )
        self.add_raw_payload(paper_id, "llm_refinement", normalized)
        row = self.conn.execute(
            "SELECT paper_id, doi, title, abstract, year, venue, url, citation_count, influential_citation_count FROM papers WHERE paper_id = ?",
            (paper_id,),
        ).fetchone()
        paper = dict(row) if row is not None else {"paper_id": paper_id, "title": ""}
        self.conn.commit()
        self.sync_vector_document(paper_id=paper_id, paper=paper, analysis=normalized, scite_tallies=None)

    def init_sqlite_vector_schema(self, embedding_model: str) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS vector_documents (
                paper_id TEXT PRIMARY KEY,
                embedding_model TEXT NOT NULL,
                document TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                indexed_at TEXT NOT NULL,
                FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
            );

            CREATE INDEX IF NOT EXISTS idx_vector_documents_model
            ON vector_documents (embedding_model);
            """
        )
        self.conn.commit()

    def sync_vector_document(
        self,
        *,
        paper_id: str,
        paper: dict[str, Any],
        analysis: dict[str, Any],
        scite_tallies: dict[str, Any] | None,
    ) -> None:
        if scite_tallies is None:
            scite_row = self.conn.execute(
                """
                SELECT scite_supporting_count, scite_contradicting_count, scite_mentioning_count, scite_total_count
                FROM papers
                WHERE paper_id = ?
                """,
                (paper_id,),
            ).fetchone()
            scite_tallies = {
                "supporting": scite_row["scite_supporting_count"] if scite_row else None,
                "contradicting": scite_row["scite_contradicting_count"] if scite_row else None,
                "mentioning": scite_row["scite_mentioning_count"] if scite_row else None,
                "total": scite_row["scite_total_count"] if scite_row else None,
            }
        vector_payload = {
            "title": paper.get("title") or "",
            "abstract": paper.get("abstract") or "",
            "task_type": analysis.get("task_type") or "",
            "framework": analysis.get("framework") or "",
            "dataset_names": json.dumps(analysis.get("dataset_names") or []),
            "venue": paper.get("venue") or "",
            "year": paper.get("year") or "",
            "doi": paper.get("doi") or (paper.get("externalIds") or {}).get("DOI") or (paper.get("externalIds") or {}).get("doi") or "",
            "code_url": analysis.get("code_url") or "",
            "scite_supporting_count": count_from_tallies(scite_tallies or {}, "supporting"),
            "scite_contradicting_count": count_from_tallies(scite_tallies or {}, "contradicting"),
            "scite_mentioning_count": count_from_tallies(scite_tallies or {}, "mentioning"),
        }
        self.conn.execute(
            """
            INSERT INTO vector_documents (paper_id, embedding_model, document, metadata_json, indexed_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
                embedding_model = excluded.embedding_model,
                document = excluded.document,
                metadata_json = excluded.metadata_json,
                indexed_at = excluded.indexed_at
            """,
            (
                paper_id,
                DEFAULT_VECTOR_EMBEDDING_MODEL,
                vector_document_text(vector_payload),
                json.dumps(
                    {
                        "paper_id": paper_id,
                        "title": paper.get("title") or "",
                        "doi": vector_payload["doi"],
                        "year": paper.get("year") or 0,
                        "venue": paper.get("venue") or "",
                        "url": paper.get("url") or "",
                        "citation_count": paper.get("citationCount") or 0,
                        "influential_citation_count": paper.get("influentialCitationCount") or 0,
                        "task_type": analysis.get("task_type") or "",
                        "framework": analysis.get("framework") or "",
                        "code_url": analysis.get("code_url") or "",
                        "dataset_names": analysis.get("dataset_names") or [],
                        "relevance_score": analysis.get("relevance_score") or 0.0,
                        "scite_supporting_count": vector_payload["scite_supporting_count"] or 0,
                        "scite_contradicting_count": vector_payload["scite_contradicting_count"] or 0,
                        "scite_mentioning_count": vector_payload["scite_mentioning_count"] or 0,
                        "scite_total_count": count_from_tallies(scite_tallies or {}, "total") or 0,
                    },
                    sort_keys=True,
                ),
                utc_now(),
            ),
        )
        self.conn.commit()

    def upsert_vector_documents(self, items: list[dict[str, Any]], embedding_model: str) -> None:
        now = utc_now()
        rows = [
            (
                item["id"],
                embedding_model,
                item["document"],
                json.dumps(item["metadata"], sort_keys=True),
                now,
            )
            for item in items
        ]
        self.conn.executemany(
            """
            INSERT INTO vector_documents (
                paper_id, embedding_model, document, metadata_json, indexed_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
                embedding_model = excluded.embedding_model,
                document = excluded.document,
                metadata_json = excluded.metadata_json,
                indexed_at = excluded.indexed_at
            """,
            rows,
        )
        self.conn.commit()

    def iter_sqlite_vectors(self, embedding_model: str) -> Iterable[dict[str, Any]]:
        sql = """
            SELECT paper_id, document, metadata_json
            FROM vector_documents
            WHERE embedding_model = ?
        """
        try:
            for row in self.conn.execute(sql, (embedding_model,)):
                yield {
                    "paper_id": row["paper_id"],
                    "document": row["document"],
                    "metadata": json.loads(row["metadata_json"]),
                }
        except sqlite3.OperationalError as exc:
            if "no such table: vector_documents" in str(exc):
                raise SystemExit("No SQLite vector index found. Run: python3 pipeline.py build-vector-index --vector-backend sqlite") from exc
            raise

    def get_scite_paper_url(self, paper_id: str) -> str | None:
        row = self.conn.execute(
            """
            SELECT raw_json
            FROM api_enrichment
            WHERE paper_id = ? AND source = 'scite_paper'
            ORDER BY fetched_at DESC
            LIMIT 1
            """,
            (paper_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(row["raw_json"])
        except (TypeError, json.JSONDecodeError):
            return None
        return extract_scite_paper_url(payload)

    def iter_vector_documents(self, *, min_relevance_score: float) -> Iterable[dict[str, Any]]:
        sql = """
            SELECT
                paper_id, doi, title, abstract, year, venue, url, citation_count,
                influential_citation_count, task_type, framework, code_url,
                dataset_names, relevance_score, scite_supporting_count,
                scite_contradicting_count, scite_mentioning_count, scite_total_count
            FROM papers
            WHERE relevance_score >= ?
            ORDER BY relevance_score DESC, citation_count DESC
        """
        for row in self.conn.execute(sql, (min_relevance_score,)):
            metadata = {
                "paper_id": row["paper_id"],
                "doi": row["doi"] or "",
                "title": row["title"] or "",
                "year": row["year"] or 0,
                "venue": row["venue"] or "",
                "url": row["url"] or "",
                "citation_count": row["citation_count"] or 0,
                "influential_citation_count": row["influential_citation_count"] or 0,
                "task_type": row["task_type"] or "",
                "framework": row["framework"] or "",
                "code_url": row["code_url"] or "",
                "dataset_names": row["dataset_names"] or "[]",
                "relevance_score": row["relevance_score"] or 0.0,
                "scite_supporting_count": row["scite_supporting_count"] or 0,
                "scite_contradicting_count": row["scite_contradicting_count"] or 0,
                "scite_mentioning_count": row["scite_mentioning_count"] or 0,
                "scite_total_count": row["scite_total_count"] or 0,
            }
            yield {
                "id": row["paper_id"],
                "document": vector_document_text(dict(row)),
                "metadata": metadata,
            }


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def detect_framework(text: str) -> str | None:
    found = []
    for needle, label in FRAMEWORK_TERMS.items():
        if needle in text and label not in found:
            found.append(label)
    return ", ".join(found) if found else None


def extract_code_url(text: str) -> str | None:
    match = re.search(r"https?://(?:www\.)?(?:github|gitlab|bitbucket)\.com/[^\s),;]+", text)
    return match.group(0) if match else None


def infer_task_type(is_cv: bool, is_yield: bool) -> str:
    if is_cv and is_yield:
        return "cv_disease_detection_and_yield_forecasting"
    if is_cv:
        return "cv_disease_detection"
    if is_yield:
        return "yield_forecasting"
    return "other"


def score_relevance(
    is_agriculture: bool,
    is_cv_disease_detection: bool,
    is_yield_forecasting: bool,
    framework: str | None,
    code_url: str | None,
) -> float:
    score = 0.0
    if is_agriculture:
        score += 0.35
    if is_cv_disease_detection:
        score += 0.3
    if is_yield_forecasting:
        score += 0.3
    if framework and ("PyTorch" in framework or "JAX" in framework):
        score += 0.15
    elif framework:
        score += 0.05
    if code_url:
        score += 0.1
    return min(score, 1.0)


def clamp_float(value: Any, fallback: float) -> float:
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return fallback


def scalar_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, sort_keys=True)


def extract_scite_paper_url(payload: dict[str, Any]) -> str | None:
    for field in ("publicationLinks", "preprintLinks"):
        links = payload.get(field) or []
        if not isinstance(links, list):
            continue
        for link in links:
            if isinstance(link, str) and link.strip():
                return link.strip()
            if isinstance(link, dict):
                for key in ("url", "href", "link"):
                    url = link.get(key)
                    if isinstance(url, str) and url.strip():
                        return url.strip()
    slug = payload.get("slug")
    if isinstance(slug, str) and slug.strip():
        return f"https://scite.ai/paper/{slug.strip()}"
    paper_id = payload.get("id")
    if paper_id is not None:
        return f"https://scite.ai/paper/{paper_id}"
    return None


def normalize_analysis_for_storage(analysis: dict[str, Any]) -> dict[str, Any]:
    dataset_names = analysis.get("dataset_names") or []
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    elif not isinstance(dataset_names, list):
        dataset_names = [dataset_names]
    return {
        "task_type": scalar_text(analysis.get("task_type")),
        "is_agriculture_related": parse_bool(analysis.get("is_agriculture_related")),
        "is_cv_disease_detection": parse_bool(analysis.get("is_cv_disease_detection")),
        "is_yield_forecasting": parse_bool(analysis.get("is_yield_forecasting")),
        "framework": scalar_text(analysis.get("framework")),
        "code_url": scalar_text(analysis.get("code_url")),
        "dataset_names": [scalar_text(item) for item in dataset_names if scalar_text(item)],
        "relevance_score": clamp_float(analysis.get("relevance_score"), 0.0),
        "rationale": scalar_text(analysis.get("rationale")),
    }


def vector_document_text(row: dict[str, Any]) -> str:
    dataset_names = row.get("dataset_names") or "[]"
    try:
        parsed_datasets = json.loads(dataset_names) if isinstance(dataset_names, str) else dataset_names
    except json.JSONDecodeError:
        parsed_datasets = dataset_names
    parts = [
        f"Title: {row.get('title') or ''}",
        f"Abstract: {row.get('abstract') or ''}",
        f"Task type: {row.get('task_type') or ''}",
        f"Framework: {row.get('framework') or ''}",
        f"Datasets: {parsed_datasets}",
        f"Venue: {row.get('venue') or ''}",
        f"Year: {row.get('year') or ''}",
        f"DOI: {row.get('doi') or ''}",
        f"Code URL: {row.get('code_url') or ''}",
        (
            "Scite: "
            f"supporting={row.get('scite_supporting_count') or 0}, "
            f"contradicting={row.get('scite_contradicting_count') or 0}, "
            f"mentioning={row.get('scite_mentioning_count') or 0}"
        ),
    ]
    return "\n".join(parts)


def normalize_query_variant(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def limit_unique_queries(queries: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for query in queries:
        normalized = normalize_query_variant(query)
        if not normalized or normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        output.append(normalized)
        if len(output) >= limit:
            break
    return output


def merge_unique_values(*values: Any) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        items = value if isinstance(value, list) else [value]
        for item in items:
            text = scalar_text(item)
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(text)
    return merged


def heuristic_prompt_variants(prompt: str, max_variants: int = 5) -> list[str]:
    base = normalize_query_variant(prompt)
    tokens = tokenize(base)
    token_text = " ".join(tokens[: min(8, len(tokens))])
    variants = [
        base,
        token_text,
        f"{base} review",
        f"{base} patterns",
        f"{base} distribution",
        f"{base} outbreak",
        f"{base} forecasting",
        f"{base} detection",
    ]
    if any(term in normalized_text(base) for term in {"locust", "grasshopper", "insect", "pest"}):
        variants.extend(
            [
                f"{base} agricultural impact",
                f"{base} Midwest United States",
                f"{base} monitoring",
            ]
        )
    return limit_unique_queries(variants, max_variants + 1)


def tokenize(value: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", normalized_text(value)) if token]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "y"}
    return False


def langchain_is_available(config: Config) -> bool:
    return bool(
        config.use_llm
        and ChatOpenAI
        and ChatPromptTemplate
        and JsonOutputParser
        and llm_api_key(config)
    )


def default_llm_model(provider: str) -> str:
    if provider == "groq":
        return "llama-3.3-70b-versatile"
    return "gpt-4.1-mini"


def llm_api_key(config: Config) -> str | None:
    if config.llm_provider == "groq":
        return os.getenv("GROQ_API_KEY")
    return os.getenv("OPENAI_API_KEY")


def build_chat_model(config: Config) -> Any:
    api_key = llm_api_key(config)
    if config.llm_provider == "groq":
        return ChatOpenAI(
            model=config.llm_model,
            temperature=0,
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
    return ChatOpenAI(model=config.llm_model, temperature=0, api_key=api_key)


def count_from_tallies(tallies: dict[str, Any], key: str) -> int | None:
    value = tallies.get(key)
    if value is None and isinstance(tallies.get("tallies"), dict):
        value = tallies["tallies"].get(key)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def dedupe_key(paper: dict[str, Any]) -> str:
    external = paper.get("externalIds") or {}
    doi = external.get("DOI") or external.get("doi")
    if doi:
        return f"doi:{doi.lower()}"
    if paper.get("paperId"):
        return f"s2:{paper['paperId']}"
    return f"title:{stable_id(normalized_text(paper.get('title') or ''))}"


def should_keep(analysis: dict[str, Any], min_relevance_score: float) -> bool:
    target_task = analysis.get("is_cv_disease_detection") or analysis.get("is_yield_forecasting")
    return bool(analysis.get("is_agriculture_related") and target_task and analysis.get("relevance_score", 0) >= min_relevance_score)


def run_pipeline(config: Config, queries: list[str]) -> None:
    http = RateLimitedClient(config.request_timeout, config.max_retries, config.base_backoff_seconds)
    semantic_scholar = SemanticScholarClient(config, http)
    scite = SciteClient(config, http)
    planner = SearchPlanner(config)
    analyzer = PaperAnalyzer(config)
    store = SQLiteStore(config.sqlite_path)
    seen_keys: set[str] = set()
    planned_queries = planner.plan(queries)

    try:
        for query in planned_queries:
            LOGGER.info("Searching Semantic Scholar: %s", query)
            run_id = store.start_run(query)
            seen = 0
            kept = 0
            try:
                try:
                    for paper in semantic_scholar.search(query):
                        seen += 1
                        key = dedupe_key(paper)
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        analysis = analyzer.analyze(paper)
                        if not should_keep(analysis, config.min_relevance_score):
                            continue

                        scite_tallies: dict[str, Any] = {}
                        scite_paper: dict[str, Any] = {}
                        doi = (paper.get("externalIds") or {}).get("DOI")
                        if doi:
                            try:
                                scite_tallies = scite.get_tallies(doi)
                                scite_paper = scite.get_paper(doi)
                            except Exception as exc:
                                LOGGER.warning("Scite enrichment failed for DOI %s: %s", doi, exc)
                        paper_id = store.upsert_paper(paper, analysis, scite_tallies, scite_paper)
                        kept += 1
                        LOGGER.info("Kept %s | %.2f | %s", paper_id, analysis["relevance_score"], paper.get("title"))
                except RuntimeError as exc:
                    LOGGER.error("Semantic Scholar query failed and will be skipped: query=%s error=%s", query, exc)
            finally:
                store.complete_run(run_id, seen, kept)
                LOGGER.info("Completed query: seen=%s kept=%s query=%s", seen, kept, query)
    finally:
        store.close()


def refine_with_llm(
    config: Config,
    *,
    limit: int | None,
    min_current_relevance_score: float,
    dry_run: bool,
    include_already_refined: bool,
) -> None:
    if not langchain_is_available(config):
        raise SystemExit(
            "LLM refinement requires langchain dependencies and an API key for the selected provider. "
            "For Groq, export GROQ_API_KEY. For OpenAI, export OPENAI_API_KEY. Do not pass --no-llm."
        )

    analyzer = PaperAnalyzer(config)
    store = SQLiteStore(config.sqlite_path)
    refined = 0
    try:
        for paper in store.iter_papers_for_refinement(
            limit=limit,
            min_current_relevance_score=min_current_relevance_score,
            include_already_refined=include_already_refined,
        ):
            paper_id = paper["_db_paper_id"]
            analysis = analyzer.analyze(paper)
            if dry_run:
                refined += 1
                LOGGER.info(
                    "Would refine %s | %.2f | %s",
                    paper_id,
                    analysis["relevance_score"],
                    paper.get("title"),
                )
            else:
                store.update_paper_analysis(paper_id, analysis)
                refined += 1
                LOGGER.info(
                    "Refined %s | %.2f | %s",
                    paper_id,
                    analysis["relevance_score"],
                    paper.get("title"),
                )
            if config.llm_delay_seconds > 0:
                time.sleep(config.llm_delay_seconds)
    finally:
        store.close()
    LOGGER.info(
        "LLM refinement complete: papers_processed=%s dry_run=%s include_already_refined=%s",
        refined,
        dry_run,
        include_already_refined,
    )


def refine_papers_by_id_with_llm(
    config: Config,
    paper_ids: list[str],
    *,
    dry_run: bool = False,
) -> int:
    if not paper_ids:
        return 0
    if not llm_api_key(config):
        LOGGER.warning("LLM API key for provider %s is not set; skipping post-crawl refinement.", config.llm_provider)
        return 0

    analyzer = PaperAnalyzer(config)
    store = SQLiteStore(config.sqlite_path)
    refined = 0
    try:
        for paper in store.iter_papers_by_ids(paper_ids):
            paper_id = paper["_db_paper_id"]
            analysis = analyzer.analyze(paper)
            if dry_run:
                LOGGER.info("Would refine %s | %.2f | %s", paper_id, analysis["relevance_score"], paper.get("title"))
            else:
                store.update_paper_analysis(paper_id, analysis)
                LOGGER.info("Refined %s | %.2f | %s", paper_id, analysis["relevance_score"], paper.get("title"))
            refined += 1
            if config.llm_delay_seconds > 0:
                time.sleep(config.llm_delay_seconds)
    finally:
        store.close()
    return refined


def require_chroma() -> None:
    if chromadb is None or embedding_functions is None:
        raise SystemExit("Vector commands require chromadb and sentence-transformers. Run: python3 -m pip install -r requirements.txt")


def require_tfidf() -> None:
    if TfidfVectorizer is None:
        raise SystemExit("SQLite vector commands require scikit-learn. Run: python3 -m pip install -r requirements.txt")


def chroma_collection(index_path: str, collection_name: str, embedding_model: str) -> Any:
    require_chroma()
    client = chromadb.PersistentClient(path=index_path)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )


def build_vector_index(
    *,
    sqlite_path: str,
    backend: str,
    index_path: str,
    collection_name: str,
    embedding_model: str,
    min_relevance_score: float,
    batch_size: int,
) -> None:
    if backend == "chroma":
        build_chroma_vector_index(
            sqlite_path=sqlite_path,
            index_path=index_path,
            collection_name=collection_name,
            embedding_model=embedding_model,
            min_relevance_score=min_relevance_score,
            batch_size=batch_size,
        )
        return

    build_sqlite_vector_index(
        sqlite_path=sqlite_path,
        embedding_model=embedding_model,
        min_relevance_score=min_relevance_score,
        batch_size=batch_size,
    )


def build_chroma_vector_index(
    *,
    sqlite_path: str,
    index_path: str,
    collection_name: str,
    embedding_model: str,
    min_relevance_score: float,
    batch_size: int,
) -> None:
    store = SQLiteStore(sqlite_path)
    collection = chroma_collection(index_path, collection_name, embedding_model)
    indexed = 0
    try:
        batch_ids: list[str] = []
        batch_documents: list[str] = []
        batch_metadatas: list[dict[str, Any]] = []
        for item in store.iter_vector_documents(min_relevance_score=min_relevance_score):
            batch_ids.append(item["id"])
            batch_documents.append(item["document"])
            batch_metadatas.append(item["metadata"])
            if len(batch_ids) >= batch_size:
                collection.upsert(ids=batch_ids, documents=batch_documents, metadatas=batch_metadatas)
                indexed += len(batch_ids)
                LOGGER.info("Indexed %s papers", indexed)
                batch_ids, batch_documents, batch_metadatas = [], [], []
        if batch_ids:
            collection.upsert(ids=batch_ids, documents=batch_documents, metadatas=batch_metadatas)
            indexed += len(batch_ids)
    finally:
        store.close()
    LOGGER.info("Vector index complete: indexed=%s path=%s collection=%s", indexed, index_path, collection_name)


def build_sqlite_vector_index(
    *,
    sqlite_path: str,
    embedding_model: str,
    min_relevance_score: float,
    batch_size: int,
) -> None:
    require_tfidf()
    store = SQLiteStore(sqlite_path)
    store.init_sqlite_vector_schema(embedding_model)
    indexed = 0
    try:
        batch: list[dict[str, Any]] = []
        for item in store.iter_vector_documents(min_relevance_score=min_relevance_score):
            batch.append(item)
            if len(batch) >= batch_size:
                store.upsert_vector_documents(batch, embedding_model)
                indexed += len(batch)
                LOGGER.info("Indexed %s papers", indexed)
                batch = []
        if batch:
            store.upsert_vector_documents(batch, embedding_model)
            indexed += len(batch)
    finally:
        store.close()
    LOGGER.info("SQLite vector index complete: indexed=%s sqlite_path=%s", indexed, sqlite_path)


def search_vector_index(
    *,
    query: str,
    backend: str,
    sqlite_path: str,
    index_path: str,
    collection_name: str,
    embedding_model: str,
    top_k: int,
    export_csv: str | None,
) -> None:
    if backend == "chroma":
        search_chroma_vector_index(
            query=query,
            index_path=index_path,
            collection_name=collection_name,
            embedding_model=embedding_model,
            top_k=top_k,
            sqlite_path=sqlite_path,
            export_csv=export_csv,
        )
        return

    search_sqlite_vector_index(
        query=query,
        sqlite_path=sqlite_path,
        embedding_model=embedding_model,
        top_k=top_k,
        export_csv=export_csv,
    )


def search_chroma_vector_index(
    *,
    query: str,
    index_path: str,
    collection_name: str,
    embedding_model: str,
    top_k: int,
    sqlite_path: str,
    export_csv: str | None,
) -> None:
    collection = chroma_collection(index_path, collection_name, embedding_model)
    result = collection.query(query_texts=[query], n_results=top_k)
    ids = result.get("ids", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    documents = result.get("documents", [[]])[0]
    ranked: list[tuple[float, dict[str, Any]]] = []
    for idx, paper_id in enumerate(ids):
        metadata = metadatas[idx] if idx < len(metadatas) else {}
        distance = distances[idx] if idx < len(distances) else None
        document = documents[idx] if idx < len(documents) else ""
        score = 1 - distance if isinstance(distance, (int, float)) else 0.0
        ranked.append(
            (
                score,
                {
                    "paper_id": paper_id,
                    "document": document,
                    "metadata": metadata,
                },
            )
        )
    rows = build_vector_search_rows(ranked=ranked, sqlite_path=sqlite_path)
    if not rows:
        raise SystemExit("No search results found.")
    print_vector_search_rows(rows)
    if export_csv:
        export_vector_search_rows_csv(rows, export_csv)


def search_sqlite_vector_index(
    *,
    query: str,
    sqlite_path: str,
    embedding_model: str,
    top_k: int,
    export_csv: str | None,
) -> None:
    ranked = score_sqlite_vector_index(query=query, sqlite_path=sqlite_path, embedding_model=embedding_model, top_k=top_k)
    if not ranked:
        raise SystemExit("No SQLite vector index found. Run: python3 pipeline.py build-vector-index")
    rows = build_vector_search_rows(ranked=ranked, sqlite_path=sqlite_path)
    print_vector_search_rows(rows)
    if export_csv:
        export_vector_search_rows_csv(rows, export_csv)


def score_sqlite_vector_index(
    *,
    query: str,
    sqlite_path: str,
    embedding_model: str,
    top_k: int,
) -> list[tuple[float, dict[str, Any]]]:
    require_tfidf()
    store = SQLiteStore(sqlite_path)
    try:
        items = list(store.iter_sqlite_vectors(embedding_model))
    finally:
        store.close()
    if not items:
        return []
    documents = [item["document"] for item in items]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=4096)
    doc_matrix = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])
    scores = (doc_matrix @ query_vector.T).toarray().ravel()
    return sorted(zip(scores, items), key=lambda pair: pair[0], reverse=True)[:top_k]


def aggregate_ranked_results(rankings: list[list[tuple[float, dict[str, Any]]]], top_k: int) -> list[tuple[float, dict[str, Any]]]:
    best_by_id: dict[str, tuple[float, dict[str, Any]]] = {}
    for ranking in rankings:
        for score, item in ranking:
            paper_id = item["paper_id"]
            current = best_by_id.get(paper_id)
            if current is None or score > current[0]:
                best_by_id[paper_id] = (score, item)
    return sorted(best_by_id.values(), key=lambda pair: pair[0], reverse=True)[:top_k]


def build_vector_search_rows(*, ranked: list[tuple[float, dict[str, Any]]], sqlite_path: str) -> list[dict[str, Any]]:
    store = SQLiteStore(sqlite_path)
    rows: list[dict[str, Any]] = []
    try:
        for rank, (score, item) in enumerate(ranked, start=1):
            metadata = item.get("metadata") or {}
            document = item.get("document") or ""
            rows.append(
                {
                    "rank": rank,
                    "score": score,
                    "paper_id": item["paper_id"],
                    "title": metadata.get("title") or "",
                    "year": metadata.get("year") or "",
                    "task_type": metadata.get("task_type") or "",
                    "framework": metadata.get("framework") or "",
                    "doi": metadata.get("doi") or "",
                    "semantic_scholar_url": metadata.get("url") or "",
                    "scite_url": store.get_scite_paper_url(item["paper_id"]) or "",
                    "snippet": first_nonempty_line(document),
                    "citation_count": metadata.get("citation_count") or 0,
                    "influential_citation_count": metadata.get("influential_citation_count") or 0,
                    "relevance_score": metadata.get("relevance_score") or 0.0,
                    "scite_supporting_count": metadata.get("scite_supporting_count") or 0,
                    "scite_contradicting_count": metadata.get("scite_contradicting_count") or 0,
                    "scite_mentioning_count": metadata.get("scite_mentioning_count") or 0,
                    "scite_total_count": metadata.get("scite_total_count") or 0,
                }
            )
    finally:
        store.close()
    return rows


def print_vector_search_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        print(f"{row['rank']}. {row['title']}")
        print(f"   paper_id: {row['paper_id']}")
        print(f"   score: {row['score']:.4f}")
        print(f"   year: {row['year']} | task: {row['task_type']} | framework: {row['framework']}")
        print(f"   doi: {row['doi']}")
        if row["scite_url"]:
            print(f"   scite: {row['scite_url']}")
        if row["semantic_scholar_url"]:
            print(f"   semantic_scholar: {row['semantic_scholar_url']}")
        print(f"   {row['snippet']}")


def export_vector_search_rows_csv(rows: list[dict[str, Any]], csv_path: str) -> None:
    directory = os.path.dirname(csv_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fieldnames = [
        "rank",
        "score",
        "paper_id",
        "title",
        "year",
        "task_type",
        "framework",
        "doi",
        "semantic_scholar_url",
        "scite_url",
        "snippet",
        "citation_count",
        "influential_citation_count",
        "relevance_score",
        "scite_supporting_count",
        "scite_contradicting_count",
        "scite_mentioning_count",
        "scite_total_count",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    LOGGER.info("Exported %s rows to %s", len(rows), csv_path)


def crawl_prompt_into_db(
    *,
    config: Config,
    query: str,
    max_results: int,
    page_size: int,
    min_relevance_score: float,
) -> int:
    http = RateLimitedClient(config.request_timeout, config.max_retries, config.base_backoff_seconds)
    scite = SciteClient(config, http)
    heuristic_config = dataclasses.replace(config, use_llm=False)
    query_analyzer = QueryRelevanceAnalyzer(heuristic_config)
    paper_analyzer = PaperAnalyzer(heuristic_config)
    temp_config = dataclasses.replace(
        config,
        max_results_per_query=max_results,
        page_size=page_size,
    )
    semantic_scholar = SemanticScholarClient(temp_config, http)
    store = SQLiteStore(config.sqlite_path)
    seen_keys: set[str] = set()
    new_ids: list[str] = []
    try:
        for paper in semantic_scholar.search(query):
            key = dedupe_key(paper)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            query_analysis = query_analyzer.analyze(query, paper)
            if not query_analysis.get("is_relevant") or query_analysis.get("relevance_score", 0) < min_relevance_score:
                continue
            paper_analysis = paper_analyzer.analyze(paper)
            scite_tallies: dict[str, Any] = {}
            scite_paper: dict[str, Any] = {}
            doi = (paper.get("externalIds") or {}).get("DOI")
            if doi:
                try:
                    scite_tallies = scite.get_tallies(doi)
                    scite_paper = scite.get_paper(doi)
                except Exception as exc:
                    LOGGER.warning("Scite enrichment failed for DOI %s: %s", doi, exc)
            ingest_analysis = {
                "task_type": paper_analysis.get("task_type") or query_analysis.get("task_type") or "query_expansion",
                "is_agriculture_related": True,
                "is_cv_disease_detection": bool(paper_analysis.get("is_cv_disease_detection")),
                "is_yield_forecasting": bool(paper_analysis.get("is_yield_forecasting")),
                "framework": paper_analysis.get("framework") or query_analysis.get("framework"),
                "code_url": paper_analysis.get("code_url") or query_analysis.get("code_url"),
                "dataset_names": merge_unique_values(query_analysis.get("dataset_names"), paper_analysis.get("dataset_names")),
                "relevance_score": max(
                    query_analysis.get("relevance_score") or 0.0,
                    paper_analysis.get("relevance_score") or 0.0,
                ),
                "rationale": f"query: {query_analysis.get('rationale')}; paper: {paper_analysis.get('rationale')}",
            }
            paper_id = store.upsert_paper(paper, ingest_analysis, scite_tallies, scite_paper)
            store.add_raw_payload(paper_id, "query_relevance", query_analysis)
            store.add_raw_payload(paper_id, "paper_analysis", paper_analysis)
            new_ids.append(paper_id)
    finally:
        store.close()
    return new_ids


def search_or_expand_vector_index(
    *,
    config: Config,
    query: str,
    sqlite_path: str,
    embedding_model: str,
    top_k: int,
    min_hits: int,
    min_score: float,
    expand_max_results: int,
    expand_page_size: int,
    prompt_variants: int,
    export_csv: str | None,
) -> None:
    queries = heuristic_prompt_variants(query, max_variants=prompt_variants)
    ranked = aggregate_ranked_results(
        [
            score_sqlite_vector_index(query=variant, sqlite_path=sqlite_path, embedding_model=embedding_model, top_k=top_k)
            for variant in queries
        ],
        top_k=top_k,
    )
    usable = [entry for entry in ranked if entry[0] >= min_score]
    if len(usable) < min_hits:
        LOGGER.info(
            "Vector index returned %s usable hits below threshold; expanding from Semantic Scholar for query: %s",
            len(usable),
            query,
        )
        newly_indexed_ids: list[str] = []
        for variant in queries:
            LOGGER.info("Expanding query variant: %s", variant)
            try:
                new_ids = crawl_prompt_into_db(
                    config=config,
                    query=variant,
                    max_results=expand_max_results,
                    page_size=expand_page_size,
                    min_relevance_score=min_score,
                )
                newly_indexed_ids.extend(new_ids)
            except Exception as exc:
                LOGGER.warning("Expansion crawl failed for query variant %s: %s", variant, exc)
        if newly_indexed_ids:
            LOGGER.info("Refining %s newly indexed papers with Groq", len(newly_indexed_ids))
            refine_papers_by_id_with_llm(config, newly_indexed_ids)
        ranked = aggregate_ranked_results(
            [
                score_sqlite_vector_index(query=variant, sqlite_path=sqlite_path, embedding_model=embedding_model, top_k=top_k)
                for variant in queries
            ],
            top_k=top_k,
        )
    if not ranked:
        raise SystemExit("No vector documents available.")
    rows = build_vector_search_rows(ranked=ranked, sqlite_path=sqlite_path)
    print_vector_search_rows(rows)
    if export_csv:
        export_vector_search_rows_csv(rows, export_csv)


def first_nonempty_line(value: str) -> str:
    for line in value.splitlines():
        line = line.strip()
        if line and not line.startswith("Title:"):
            return line[:220]
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a SQLite dataset of agriculture AI papers.")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["crawl", "refine-with-llm", "build-vector-index", "search-vector", "search-or-expand"],
        default="crawl",
    )
    parser.add_argument("vector_query", nargs="?", help="Semantic search query for search-vector.")
    parser.add_argument("--sqlite-path", default="sciteai_papers.sqlite3")
    parser.add_argument("--query", action="append", help="Additional search query. Can be repeated.")
    parser.add_argument("--queries-file", help="Text file with one query per line.")
    parser.add_argument("--no-default-queries", action="store_true", help="Run only queries supplied with --query or --queries-file.")
    parser.add_argument("--max-results-per-query", type=int, default=100)
    parser.add_argument("--page-size", type=int, default=25)
    parser.add_argument("--semantic-scholar-start-offset", type=int, default=0)
    parser.add_argument("--min-relevance-score", type=float, default=0.55)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--base-backoff-seconds", type=float, default=1.0)
    parser.add_argument("--semantic-scholar-base-url", default="https://api.semanticscholar.org/graph/v1")
    parser.add_argument("--scite-base-url", default="https://api.scite.ai")
    parser.add_argument("--semantic-scholar-delay-seconds", type=float, default=1.0)
    parser.add_argument("--llm-provider", choices=["openai", "groq"], default="groq")
    parser.add_argument("--llm-model", help="LLM model name. Defaults to provider-specific model.")
    parser.add_argument("--llm-delay-seconds", type=float, default=0.0, help="Delay between LLM refinement calls.")
    parser.add_argument("--no-llm", action="store_true", help="Use deterministic keyword analysis only.")
    parser.add_argument("--refine-limit", type=int, help="Maximum saved papers to refine with the LLM.")
    parser.add_argument("--refine-min-current-score", type=float, default=0.0, help="Only refine papers at or above this current relevance score.")
    parser.add_argument("--include-already-refined", action="store_true", help="For refine-with-llm, include papers that already have llm_refinement payloads.")
    parser.add_argument("--dry-run", action="store_true", help="For refine-with-llm, analyze papers but do not update SQLite.")
    parser.add_argument("--vector-index-path", default="chroma_index")
    parser.add_argument("--vector-backend", choices=["sqlite", "chroma"], default="sqlite")
    parser.add_argument("--vector-collection", default="sciteai_papers")
    parser.add_argument("--embedding-model", default="tfidf-bigrams")
    parser.add_argument("--vector-min-relevance-score", type=float, default=0.0)
    parser.add_argument("--vector-batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-vector-hits", type=int, default=5)
    parser.add_argument("--min-vector-score", type=float, default=0.15)
    parser.add_argument("--export-csv", help="Write vector search results to a CSV file.")
    parser.add_argument("--expand-max-results", type=int, default=100)
    parser.add_argument("--expand-page-size", type=int, default=25)
    parser.add_argument("--prompt-variants", type=int, default=5)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_queries(args: argparse.Namespace) -> list[str]:
    queries = [] if args.no_default_queries else list(DEFAULT_QUERIES)
    if args.queries_file:
        with open(args.queries_file, "r", encoding="utf-8") as handle:
            queries.extend(line.strip() for line in handle if line.strip() and not line.startswith("#"))
    if args.query:
        queries.extend(args.query)
    deduped_queries = list(dict.fromkeys(queries))
    if not deduped_queries:
        raise SystemExit("No queries configured. Provide --query, --queries-file, or remove --no-default-queries.")
    return deduped_queries


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = Config.from_env(args)
    if args.command == "refine-with-llm":
        LOGGER.info("LLM provider: %s model: %s", config.llm_provider, config.llm_model)
        if not llm_api_key(config):
            LOGGER.warning("LLM API key for provider %s is not set.", config.llm_provider)
        refine_with_llm(
            config,
            limit=args.refine_limit,
            min_current_relevance_score=args.refine_min_current_score,
            dry_run=args.dry_run,
            include_already_refined=args.include_already_refined,
        )
    elif args.command == "build-vector-index":
        build_vector_index(
            sqlite_path=args.sqlite_path,
            backend=args.vector_backend,
            index_path=args.vector_index_path,
            collection_name=args.vector_collection,
            embedding_model=args.embedding_model,
            min_relevance_score=args.vector_min_relevance_score,
            batch_size=args.vector_batch_size,
        )
    elif args.command == "search-vector":
        if not args.vector_query:
            raise SystemExit("search-vector requires a query string.")
        search_vector_index(
            query=args.vector_query,
            backend=args.vector_backend,
            sqlite_path=args.sqlite_path,
            index_path=args.vector_index_path,
            collection_name=args.vector_collection,
            embedding_model=args.embedding_model,
            top_k=args.top_k,
            export_csv=args.export_csv,
        )
    elif args.command == "search-or-expand":
        if not args.vector_query:
            raise SystemExit("search-or-expand requires a query string.")
        search_or_expand_vector_index(
            config=config,
            query=args.vector_query,
            sqlite_path=args.sqlite_path,
            embedding_model=args.embedding_model,
            top_k=args.top_k,
            min_hits=args.min_vector_hits,
            min_score=args.min_vector_score,
            expand_max_results=args.expand_max_results,
            expand_page_size=args.expand_page_size,
            prompt_variants=args.prompt_variants,
            export_csv=args.export_csv,
        )
    else:
        LOGGER.info("Semantic Scholar API key configured: %s", bool(config.semantic_scholar_api_key))
        LOGGER.info("Scite API key configured: %s", bool(config.scite_api_key))
        LOGGER.info("LLM provider: %s model: %s", config.llm_provider, config.llm_model)
        if config.use_llm and not llm_api_key(config):
            LOGGER.warning("LLM API key for provider %s is not set; falling back to keyword analysis for crawl.", config.llm_provider)
        run_pipeline(config, load_queries(args))


if __name__ == "__main__":
    main()

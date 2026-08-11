#!/usr/bin/env python3
"""
CLI for retrieval-augmented Q&A over the indexed Wikipedia Movie Plots
ChromaDB collection (built by Rag_Pipeline.ipynb).

Three-stage pipeline:
  1. PLAN   — Gemini reads the user's raw question (query_planner_prompt.md)
              and returns a structured retrieval plan: a rewritten semantic
              search string, plus optional metadata filters (title / genre /
              director / origin / year / year_min / year_max) drawn only from
              what the question actually implies. If planning fails for any
              reason, this falls back to an unfiltered plan rather than aborting.
  2. RETRIEVE — embed the planned search string with BGE-small and query
              ChromaDB using whatever `where` clause the plan produced. If
              that filtered search comes back empty, it is automatically
              retried once with no filter at all.
  3. ANSWER — the retrieved chunks are fed into prompt_template.md and
              Gemini produces the final structured JSON answer
              ({answer, contexts, reasoning}), which is validated before
              being printed.

Setup:
    pip install chromadb sentence-transformers google-genai pydantic python-dotenv
    cp .env.example .env   # then paste your real key into .env
        GEMINI_API_KEY=...

Usage:
    python query_gemini.py "What movie features an AI that turns against its crew?"
    python query_gemini.py "What happens in Titanic?" --n-results 5
    python query_gemini.py "..." --genre "drama"   # manual override, forces this genre filter ($contains)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import chromadb
from dotenv import load_dotenv
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

try:
    from google import genai
    from google.genai import types
except ImportError:
    sys.exit("Missing dependency: run `pip install google-genai` (Gemini API SDK).")

# Resolved relative to this script's own location (same folder as Rag_Pipeline.ipynb),
# so the CLI finds the saved DB regardless of the caller's working directory.
CHROMA_PATH = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "movie_plots"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
GEMINI_MODEL = "gemini-3.5-flash-lite"
PLANNER_PROMPT_PATH = Path(__file__).parent / "query_planner_prompt.md"
ANSWER_PROMPT_PATH = Path(__file__).parent / "prompt_template.md"

# bge is asymmetric: only queries get this instruction prefix, stored passages don't
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class RetrievalPlan(BaseModel):
    search_query: str
    title: Optional[str] = None
    genre: Optional[str] = None
    director: Optional[str] = None
    origin: Optional[str] = None
    year: Optional[int] = None
    year_min: Optional[int] = None
    year_max: Optional[int] = None


class RagResponse(BaseModel):
    answer: str
    contexts: List[str]
    reasoning: str


# ---------------------------------------------------------------------------
# Stage 1: plan the retrieval query
# ---------------------------------------------------------------------------

def plan_retrieval(user_query: str, api_key: str) -> RetrievalPlan:
    """Ask Gemini to turn the raw question into a search string + metadata filters."""
    template = PLANNER_PROMPT_PATH.read_text(encoding="utf-8")
    prompt = template.replace("{{QUESTION}}", user_query)

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RetrievalPlan,
        ),
    )
    if response.parsed is not None:
        return response.parsed
    return RetrievalPlan.model_validate_json(response.text)


def build_where(plan: RetrievalPlan, genre_override: Optional[str]) -> Optional[dict]:
    """Translate a RetrievalPlan into a Chroma `where` clause, or None if no filters apply.

    `genre`/`director`/`origin` are stored as *lists* per chunk (see Rag_Pipeline.ipynb
    Step 3), so membership is checked with `$contains` rather than `$eq`. `title`/`year`
    stay scalar, so they use `$eq`/`$gte`/`$lte` as before.
    """
    conditions = []

    genre_value = genre_override or plan.genre
    if genre_value:
        conditions.append({"genre": {"$contains": genre_value}})
    if plan.title:
        conditions.append({"title": {"$eq": plan.title}})
    if plan.director:
        conditions.append({"director": {"$contains": plan.director}})
    if plan.origin:
        conditions.append({"origin": {"$contains": plan.origin}})
    if plan.year is not None:
        conditions.append({"year": {"$eq": plan.year}})
    else:
        if plan.year_min is not None:
            conditions.append({"year": {"$gte": plan.year_min}})
        if plan.year_max is not None:
            conditions.append({"year": {"$lte": plan.year_max}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ---------------------------------------------------------------------------
# Stage 2: retrieve from ChromaDB
# ---------------------------------------------------------------------------

def retrieve(search_text: str, n_results: int, where: Optional[dict], embed_model, collection):
    query_vec = embed_model.encode(
        QUERY_INSTRUCTION + search_text,
        normalize_embeddings=True,
    ).tolist()
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=n_results,
        where=where,
    )
    return results["documents"][0], results["metadatas"][0]


# ---------------------------------------------------------------------------
# Stage 3: generate the final structured answer
# ---------------------------------------------------------------------------

def build_prompt(query: str, docs: List[str], metas: List[dict]) -> str:
    template = ANSWER_PROMPT_PATH.read_text(encoding="utf-8")
    context_block = "\n\n".join(
        f"[{i + 1}] {m['title']} ({m['year']}):\n{doc}"
        for i, (doc, m) in enumerate(zip(docs, metas))
    )
    return template.replace("{{CONTEXT}}", context_block).replace("{{QUESTION}}", query)


def ask_gemini(prompt: str, api_key: str) -> RagResponse:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RagResponse,
        ),
    )
    if response.parsed is not None:
        return response.parsed
    return RagResponse.model_validate_json(response.text)


def validate_response(result: RagResponse) -> RagResponse:
    """Extra semantic validation beyond the pydantic shape check, before this reaches the user."""
    if not result.answer or not result.answer.strip():
        raise ValueError("model returned an empty 'answer'")
    if not isinstance(result.contexts, list) or not all(isinstance(c, str) for c in result.contexts):
        raise ValueError("'contexts' must be a list of strings")
    if not result.reasoning or not result.reasoning.strip():
        raise ValueError("model returned an empty 'reasoning'")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="RAG query CLI over the movie_plots ChromaDB collection, answered by Gemini."
    )
    parser.add_argument("query", help="Natural language question about the indexed movie plots.")
    parser.add_argument("--n-results", type=int, default=3, help="Chunks to retrieve (default: 3).")
    parser.add_argument(
        "--genre", default=None,
        help="Manual override: force a genre filter (e.g. 'drama') via $contains against the "
             "stored genre list, bypassing whatever genre the planner would have inferred.",
    )
    parser.add_argument("--chroma-path", default=CHROMA_PATH, help="Path to the persistent Chroma DB.")
    parser.add_argument("--api-key", default=None, help="Gemini API key (defaults to $GEMINI_API_KEY from .env).")
    args = parser.parse_args()

    load_dotenv()  # reads .env in the current/project directory
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("No API key found. Put GEMINI_API_KEY=... in a .env file, or pass --api-key.")

    chroma_client = chromadb.PersistentClient(path=args.chroma_path)
    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
    except Exception:
        sys.exit(
            f"Collection '{COLLECTION_NAME}' not found at {args.chroma_path}. "
            "Run Rag_Pipeline.ipynb first to build it."
        )

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    # --- Stage 1: plan the query (metadata filters + rewritten search text) ---
    try:
        plan = plan_retrieval(args.query, api_key)
        print(f"[planner] search_query={plan.search_query!r}", file=sys.stderr)
    except Exception as exc:
        print(f"[planner] query planning failed ({exc}); falling back to unfiltered search.", file=sys.stderr)
        plan = RetrievalPlan(search_query=args.query)

    where = build_where(plan, args.genre)
    print(f"[planner] where={where}", file=sys.stderr)

    # --- Stage 2: retrieve, with automatic unfiltered retry on empty results ---
    docs, metas = retrieve(plan.search_query, args.n_results, where, embed_model, collection)
    if not docs and where is not None:
        print("[retrieval] filtered search returned no results; retrying without filters.", file=sys.stderr)
        docs, metas = retrieve(plan.search_query, args.n_results, None, embed_model, collection)

    if not docs:
        sys.exit("No chunks retrieved even without filters — is the collection populated?")

    # --- Stage 3: generate and validate the final structured answer ---
    prompt = build_prompt(args.query, docs, metas)
    try:
        result = ask_gemini(prompt, api_key)
        result = validate_response(result)
    except Exception as exc:
        sys.exit(f"Failed to generate a valid answer: {exc}")

    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()

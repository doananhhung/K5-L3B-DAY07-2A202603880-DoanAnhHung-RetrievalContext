"""Evaluate multiple retrieval and context packing configurations on dev or held-out test split."""
import argparse
from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
import statistics
import time

from .config import (
    CHUNKS_PATH,
    DEFAULT_BUDGET_EXPANDED,
    DEFAULT_TOP_K,
    QUESTIONS_PATH,
    RESULTS_DIR,
    load_settings,
)
from .context_packing import pack_context_fifo, pack_context_priority
from .providers import OpenRouterChatClient, OpenRouterEmbedder
from .retrievers import (
    Bm25Retriever,
    HybridRrfRetriever,
    NeighborIndex,
    PgVectorDenseRetriever,
    TfidfRetriever,
    infer_safe_document_type_filter,
    load_chunks,
)
from .types import ChunkRecord, FilterMap, GoldSpan, QuestionRecord, RetrievedChunk, UsageAccumulator

SearchCallable = Callable[[QuestionRecord, FilterMap | None], list[RetrievedChunk]]


@dataclass(frozen=True)
class MethodSpec:
    """Configuration spec for one retrieval & context packing run."""

    search_fn: SearchCallable
    budget: int
    pack_mode: str
    use_safe_inferred_filter: bool
    uses_chat: bool


def is_evidence_hit(chunk: ChunkRecord | RetrievedChunk, gold: list[GoldSpan]) -> bool:
    """Return True if chunk matches source_id, page, and contains exact gold evidence."""
    return any(
        chunk["source_id"] == g["source_id"]
        and int(chunk["page"]) == int(g["page"])
        and g["evidence"] in chunk["text"]
        for g in gold
    )


def _load_questions_for_split(path: Path, split: str) -> list[QuestionRecord]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Questions file must be JSON array: {path}")
    out: list[QuestionRecord] = []
    for item in raw:
        if not isinstance(item, dict) or item.get("split") != split:
            continue
        gold_raw = item.get("gold")
        if not isinstance(gold_raw, list) or not gold_raw:
            raise ValueError(f"Question {item.get('id')} missing gold list")
        gold_spans: list[GoldSpan] = [
            GoldSpan(
                source_id=str(g["source_id"]),
                page=int(g["page"]),
                evidence=str(g["evidence"]),
            )
            for g in gold_raw
        ]
        rec: QuestionRecord = {
            "id": str(item["id"]),
            "split": "dev" if split == "dev" else "test",
            "type": str(item.get("type", "unknown")),
            "question": str(item["question"]),
            "expected_answer": str(item["expected_answer"]),
            "gold": gold_spans,
        }
        if isinstance(item.get("filter"), dict):
            rec["filter"] = {str(k): v for k, v in item["filter"].items()}
        out.append(rec)
    if not out:
        raise ValueError(f"No questions found for split '{split}' in {path}")
    return out


def _evaluate_one_method(
    method_label: str,
    split: str,
    questions: list[QuestionRecord],
    spec: MethodSpec,
    neighbor_index: NeighborIndex,
    embedder: OpenRouterEmbedder,
    chat_client: OpenRouterChatClient | None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    emb_before = embedder.usage.snapshot()
    chat_before = chat_client.usage.snapshot() if chat_client is not None else None

    traces: list[dict[str, object]] = []
    rr_values: list[float] = []
    ctx_char_values: list[int] = []
    lat_values: list[float] = []

    for q in questions:
        active_filter = infer_safe_document_type_filter(q["question"]) if spec.use_safe_inferred_filter else None

        t0 = time.perf_counter()
        results = spec.search_fn(q, active_filter)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        top3 = results[:3]
        if spec.pack_mode == "fifo":
            context, context_ids = pack_context_fifo(top3, budget=spec.budget)
        elif spec.pack_mode == "priority_heading":
            context, context_ids = pack_context_priority(
                top3,
                budget=spec.budget,
                neighbor_index=neighbor_index,
                include_heading_hint=True,
            )
        else:
            context, context_ids = pack_context_priority(
                top3,
                budget=spec.budget,
                neighbor_index=neighbor_index,
                include_heading_hint=False,
            )

        gold = q["gold"]
        marked: list[dict[str, object]] = []
        hit_flags: list[bool] = []
        ctx_hit_flags: list[bool] = []
        for c in top3:
            h = is_evidence_hit(c, gold)
            hit_flags.append(h)
            ctx_hit_flags.append(h and c["id"] in context_ids)
            marked.append({**c, "hit": h})

        first_hit_rank = next((idx for idx, flag in enumerate(hit_flags, start=1) if flag), None)
        rr = (1.0 / first_hit_rank) if first_hit_rank is not None else 0.0
        rr_values.append(rr)
        ctx_char_values.append(len(context))
        lat_values.append(elapsed_ms)

        traces.append(
            {
                "query_id": q["id"],
                "split": split,
                "type": q["type"],
                "question": q["question"],
                "method": method_label,
                "filter_used": active_filter,
                "budget_chars": spec.budget,
                "pack_mode": spec.pack_mode,
                "results": marked,
                "context": context,
                "context_ids": context_ids,
                "context_chars": len(context),
                "elapsed_ms": round(elapsed_ms, 2),
                "hit1": bool(hit_flags and hit_flags[0]),
                "hit3": any(hit_flags),
                "context_hit": any(ctx_hit_flags),
                "rr3": rr,
            }
        )

    emb_after = embedder.usage.snapshot()
    chat_after = chat_client.usage.snapshot() if chat_client is not None else None
    emb_delta = UsageAccumulator.delta(emb_before, emb_after)
    chat_delta = (
        UsageAccumulator.delta(chat_before, chat_after)
        if chat_before is not None and chat_after is not None
        else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0, "api_calls": 0}
    )
    emb_cost = emb_after[3] - emb_before[3]
    chat_cost = (chat_after[3] - chat_before[3]) if (chat_before is not None and chat_after is not None) else 0.0
    total_cost = round(emb_cost + chat_cost, 8)

    n = len(traces)
    h1 = sum(1 for t in traces if t["hit1"])
    h3 = sum(1 for t in traces if t["hit3"])
    ch = sum(1 for t in traces if t["context_hit"])
    summary: dict[str, object] = {
        "config": method_label,
        "split": split,
        "n": n,
        "budget_chars": spec.budget,
        "pack_mode": spec.pack_mode,
        "hit1": f"{h1}/{n}",
        "hit1_rate": round(h1 / n, 4),
        "hit3": f"{h3}/{n}",
        "hit3_rate": round(h3 / n, 4),
        "context_hit": f"{ch}/{n}",
        "context_hit_rate": round(ch / n, 4),
        "mrr3": round(statistics.mean(rr_values), 4),
        "avg_context_chars": round(statistics.mean(ctx_char_values), 1),
        "median_ms": round(statistics.median(lat_values), 2),
        "embed_usage": emb_delta,
        "chat_usage": chat_delta,
        "total_cost_usd": total_cost,
    }
    return summary, traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark retrieval methods on Day 7 split")
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "tfidf_b2400",
            "dense_b2400",
            "bm25_b2400",
            "bm25_ctx_b2400",
            "hybrid_rrf_b2400",
            "hybrid_rrf_filtered_b2400",
        ],
    )
    args = parser.parse_args()

    out_dir = args.output or (RESULTS_DIR / f"retrieval-{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = load_settings()
    chunks = load_chunks(args.chunks)
    questions = _load_questions_for_split(args.questions, args.split)
    neighbor_index = NeighborIndex(chunks)

    tfidf = TfidfRetriever(chunks)
    bm25 = Bm25Retriever(chunks)
    bm25_ctx = Bm25Retriever(chunks, neighbor_index=neighbor_index)
    embedder = OpenRouterEmbedder(
        model_name=settings.embedding_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )
    dense = PgVectorDenseRetriever(
        database_url=settings.database_url,
        strategy=settings.day6_strategy,
        embedder=embedder,
    )
    hybrid = HybridRrfRetriever(dense=dense, lexical=bm25)
    hybrid_ctx = HybridRrfRetriever(dense=dense, lexical=bm25_ctx)

    emb_q_before = embedder.usage.snapshot()
    t_q0 = time.perf_counter()
    query_vecs: dict[str, list[float]] = {
        q["id"]: embedder.embed_query(q["question"]) for q in questions
    }
    embed_q_seconds = round(time.perf_counter() - t_q0, 3)
    emb_q_delta = UsageAccumulator.delta(emb_q_before, embedder.usage.snapshot())

    def _run_tfidf(q: QuestionRecord, flt: FilterMap | None) -> list[RetrievedChunk]:
        return tfidf.search(q["question"], filters=flt, top_k=DEFAULT_TOP_K)

    def _run_bm25(q: QuestionRecord, flt: FilterMap | None) -> list[RetrievedChunk]:
        return bm25.search(q["question"], filters=flt, top_k=DEFAULT_TOP_K)

    def _run_bm25_ctx(q: QuestionRecord, flt: FilterMap | None) -> list[RetrievedChunk]:
        return bm25_ctx.search(q["question"], filters=flt, top_k=DEFAULT_TOP_K)

    def _run_dense(q: QuestionRecord, flt: FilterMap | None) -> list[RetrievedChunk]:
        return dense.search_by_vector(query_vecs[q["id"]], filters=flt, top_k=DEFAULT_TOP_K)

    def _run_hybrid(q: QuestionRecord, flt: FilterMap | None) -> list[RetrievedChunk]:
        return hybrid.search_with_vector(q["question"], query_vecs[q["id"]], filters=flt, top_k=DEFAULT_TOP_K)

    def _run_hybrid_ctx(q: QuestionRecord, flt: FilterMap | None) -> list[RetrievedChunk]:
        return hybrid_ctx.search_with_vector(q["question"], query_vecs[q["id"]], filters=flt, top_k=DEFAULT_TOP_K)

    specs: dict[str, MethodSpec] = {
        "tfidf_b2400": MethodSpec(_run_tfidf, DEFAULT_BUDGET_EXPANDED, "priority", False, False),
        "dense_b2400": MethodSpec(_run_dense, DEFAULT_BUDGET_EXPANDED, "priority", False, False),
        "bm25_b2400": MethodSpec(_run_bm25, DEFAULT_BUDGET_EXPANDED, "priority", False, False),
        "bm25_ctx_b2400": MethodSpec(_run_bm25_ctx, DEFAULT_BUDGET_EXPANDED, "priority_heading", True, False),
        "hybrid_rrf_b2400": MethodSpec(_run_hybrid, DEFAULT_BUDGET_EXPANDED, "priority", False, False),
        "hybrid_rrf_filtered_b2400": MethodSpec(_run_hybrid, DEFAULT_BUDGET_EXPANDED, "priority_heading", True, False),
        "hybrid_ctx_rrf_filtered_b2400": MethodSpec(_run_hybrid_ctx, DEFAULT_BUDGET_EXPANDED, "priority_heading", True, False),
    }

    chat_client: OpenRouterChatClient | None = None

    summaries: list[dict[str, object]] = []
    for name in args.methods:
        if name not in specs:
            raise ValueError(f"Unknown retrieval method '{name}'. Valid: {sorted(specs.keys())}")
        spec = specs[name]
        summary, traces = _evaluate_one_method(
            method_label=name,
            split=args.split,
            questions=questions,
            spec=spec,
            neighbor_index=neighbor_index,
            embedder=embedder,
            chat_client=chat_client if spec.uses_chat else None,
        )
        summaries.append(summary)
        (out_dir / f"traces_{name}.json").write_text(
            json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    report = {
        "split": args.split,
        "n_questions": len(questions),
        "shared_query_embedding": {
            "model": settings.embedding_model,
            "elapsed_seconds": embed_q_seconds,
            **emb_q_delta,
        },
        "methods": summaries,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

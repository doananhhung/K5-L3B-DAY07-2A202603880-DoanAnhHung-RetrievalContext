"""Run real Agentic RAG evaluations (baseline and improved configs) on dev or held-out test split."""
import argparse
import json
from pathlib import Path
import sys
import time

from .agent import (
    BaselineOpenRouterController,
    ImprovedAgentConfig,
    run_improved_agent,
)
from .config import (
    AGENT_CASES_PATH,
    CHUNKS_PATH,
    DEFAULT_BUDGET_EXPANDED,
    DEFAULT_MAX_SEARCHES,
    DEFAULT_TOP_K,
    RESULTS_DIR,
    REPO_ROOT,
    load_settings,
)
from .providers import OpenRouterChatClient, OpenRouterEmbedder
from .retrievers import (
    Bm25Retriever,
    HybridRrfRetriever,
    NeighborIndex,
    PgVectorDenseRetriever,
    RetrieverProtocol,
    load_chunks,
)
from .types import AgentCaseRecord, UsageAccumulator

# Import starter/agentic.py without modifying starter files.
STARTER_DIR = str(REPO_ROOT / "starter")
if STARTER_DIR not in sys.path:
    sys.path.insert(0, STARTER_DIR)
from agentic import run_agent as run_starter_agent  # noqa: E402


def _load_cases_for_split(path: Path, split: str) -> list[AgentCaseRecord]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Agent cases file must be JSON array: {path}")
    cases: list[AgentCaseRecord] = []
    for item in raw:
        if isinstance(item, dict) and item.get("split") == split:
            cases.append(item)  # type: ignore[arg-type]
    if not cases:
        raise ValueError(f"No agent cases found for split '{split}' in {path}")
    return cases


def _run_baseline_agent_case(
    case: AgentCaseRecord,
    chat_client: OpenRouterChatClient,
    dense_retriever: PgVectorDenseRetriever,
    embedder: OpenRouterEmbedder,
) -> dict[str, object]:
    t0 = time.perf_counter()
    c_before = chat_client.usage.snapshot()
    e_before = embedder.usage.snapshot()
    controller = BaselineOpenRouterController(chat_client=chat_client)
    try:
        res = run_starter_agent(
            case["question"],
            controller,  # type: ignore[arg-type]
            dense_retriever,  # type: ignore[arg-type]
            filters=None,
            max_searches=DEFAULT_MAX_SEARCHES,
            top_k=DEFAULT_TOP_K,
            budget=DEFAULT_BUDGET_EXPANDED,
        )
        events = res.get("events", [])
        final_ctx = ""
        final_ids: list[str] = []
        for ev in events:
            if isinstance(ev, dict) and ev.get("event") == "search":
                final_ctx = str(ev.get("context", ""))
                raw_ids = ev.get("context_ids", [])
                if isinstance(raw_ids, list):
                    final_ids = [str(x) for x in raw_ids]
        out: dict[str, object] = {
            "query_id": case["id"],
            "question": case["question"],
            "status": str(res["status"]),
            "answer": str(res["answer"]),
            "citations": list(res["citations"]),
            "events": events,
            "final_context": final_ctx,
            "final_context_ids": final_ids,
        }
    except Exception as exc:
        out = {
            "query_id": case["id"],
            "question": case["question"],
            "status": "error",
            "answer": "",
            "citations": [],
            "events": [],
            "final_context": "",
            "final_context_ids": [],
            "error": f"{type(exc).__name__}: {exc}",
        }

    out["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    out["chat_usage"] = UsageAccumulator.delta(c_before, chat_client.usage.snapshot())
    out["embed_usage"] = UsageAccumulator.delta(e_before, embedder.usage.snapshot())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real Agentic RAG evaluation on dev or test split")
    parser.add_argument(
        "--config",
        required=True,
        choices=(
            "baseline_dense_b2400",
            "c1_hybrid_b2400",
            "c2_agentic_tools_b3500",
        ),
    )
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--cases", type=Path, default=AGENT_CASES_PATH)
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.output or (RESULTS_DIR / f"agent-{args.config}-{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = load_settings()
    chunks = load_chunks(args.chunks)
    cases = _load_cases_for_split(args.cases, args.split)
    neighbor_index = NeighborIndex(chunks)

    embedder = OpenRouterEmbedder(
        model_name=settings.embedding_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )
    chat_client = OpenRouterChatClient(
        model_name=settings.chat_model,
        api_key=settings.chat_api_key,
        base_url=settings.chat_base_url,
    )
    dense = PgVectorDenseRetriever(
        database_url=settings.database_url,
        strategy=settings.day6_strategy,
        embedder=embedder,
    )
    bm25 = Bm25Retriever(chunks)
    bm25_ctx = Bm25Retriever(chunks, neighbor_index=neighbor_index)
    hybrid = HybridRrfRetriever(dense=dense, lexical=bm25)
    hybrid_ctx = HybridRrfRetriever(dense=dense, lexical=bm25_ctx)

    retrievers: dict[str, RetrieverProtocol] = {
        "dense_pgvector": dense,
        "bm25": bm25,
        "bm25_contextual": bm25_ctx,
        "hybrid_rrf": hybrid,
        "hybrid_contextual_rrf": hybrid_ctx,
    }

    configs: dict[str, ImprovedAgentConfig] = {
        "c1_hybrid_b2400": ImprovedAgentConfig(
            name="c1_hybrid_b2400",
            default_method="hybrid_rrf",
            allow_keyword_tool=False,
            allow_neighbor_tool=False,
            allow_model_filter=False,
            top_k=DEFAULT_TOP_K,
            budget=DEFAULT_BUDGET_EXPANDED,
            max_searches=3,
            pack_mode="priority",
            include_heading_hint=False,
        ),
        "c2_agentic_tools_b3500": ImprovedAgentConfig(
            name="c2_agentic_tools_b3500",
            default_method="hybrid_rrf",
            allow_keyword_tool=True,
            allow_neighbor_tool=True,
            allow_model_filter=False,
            use_safe_rule_filter=True,
            top_k=DEFAULT_TOP_K,
            budget=3500,
            max_searches=3,
            pack_mode="priority",
            include_heading_hint=True,
        ),
    }

    traces: list[dict[str, object]] = []
    t_all = time.perf_counter()
    for case in cases:
        if args.config == "baseline_dense_b2400":
            tr = _run_baseline_agent_case(case, chat_client, dense, embedder)
        else:
            cfg = configs[args.config]
            try:
                tr = run_improved_agent(
                    query_id=case["id"],
                    question=case["question"],
                    chat_client=chat_client,
                    retrievers=retrievers,
                    neighbor_index=neighbor_index,
                    config=cfg,
                    embedder=embedder,
                    preset_filters=None,
                )
            except Exception as exc:
                tr = {
                    "query_id": case["id"],
                    "question": case["question"],
                    "status": "error",
                    "answer": "",
                    "citations": [],
                    "events": [],
                    "final_context": "",
                    "final_context_ids": [],
                    "elapsed_ms": 0.0,
                    "chat_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0, "api_calls": 0},
                    "embed_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0, "api_calls": 0},
                    "error": f"{type(exc).__name__}: {exc}",
                }
        traces.append(tr)
        print(
            f"[{args.config} | {args.split}] {case['id']} -> status={tr['status']}, "
            f"searches={sum(1 for e in tr['events'] if isinstance(e, dict) and e.get('event') == 'search')}, "
            f"cits={tr['citations']}, ms={tr['elapsed_ms']}",
            flush=True,
        )

    total_seconds = round(time.perf_counter() - t_all, 2)
    run_meta = {
        "config": args.config,
        "split": args.split,
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
        "day6_strategy": settings.day6_strategy,
        "total_cases": len(cases),
        "status_counts": {
            "answered": sum(1 for t in traces if t["status"] == "answered"),
            "abstain": sum(1 for t in traces if t["status"] == "abstain"),
            "error": sum(1 for t in traces if t["status"] == "error"),
        },
        "total_seconds": total_seconds,
        "chat_usage": {
            "prompt_tokens": chat_client.usage.prompt_tokens,
            "completion_tokens": chat_client.usage.completion_tokens,
            "total_tokens": chat_client.usage.total_tokens,
            "cost_usd": round(chat_client.usage.cost_usd, 8),
            "api_calls": chat_client.usage.api_calls,
        },
        "embed_usage": {
            "prompt_tokens": embedder.usage.prompt_tokens,
            "total_tokens": embedder.usage.total_tokens,
            "cost_usd": round(embedder.usage.cost_usd, 8),
            "api_calls": embedder.usage.api_calls,
        },
        "total_cost_usd": round(chat_client.usage.cost_usd + embedder.usage.cost_usd, 8),
    }

    (out_dir / "traces.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "run_meta.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

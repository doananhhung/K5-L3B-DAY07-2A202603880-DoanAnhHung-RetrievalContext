"""Unit tests for Day 7 custom retrievers, context packing, neighbor navigation, and filter guards."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.context_packing import pack_context_fifo, pack_context_priority  # noqa: E402
from src.providers import _extract_usage_fields, extract_json_object  # noqa: E402
from src.retrievers import (  # noqa: E402
    Bm25Retriever,
    HybridRrfRetriever,
    LlmRerankRetriever,
    NeighborIndex,
    TfidfRetriever,
    infer_safe_document_type_filter,
    validate_filters,
)
from src.types import ChatCompletionResult, ChunkRecord, RetrievedChunk  # noqa: E402


def _make_chunk(cid: str, page: int, idx: int, text: str, doc_type: str = "luat") -> ChunkRecord:
    return ChunkRecord(
        id=cid,
        source_id="luat-p1",
        source="luat.pdf",
        page=page,
        text=text,
        metadata={"document_type": doc_type, "chunk_index": idx},
    )


class _StubDenseRetriever:
    def __init__(self, hits: list[RetrievedChunk]) -> None:
        self._hits = hits

    def search(self, query: str, *, filters=None, top_k: int = 3) -> list[RetrievedChunk]:
        return self._hits[:top_k]


class _StubChatClient:
    def __init__(self, response_json: str) -> None:
        self._response_json = response_json

    def complete(self, messages, tools=None, temperature: float = 0.0, max_tokens: int = 1024, json_mode: bool = False):
        return ChatCompletionResult(content=self._response_json)


class Day7PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            _make_chunk("luat-p1:p007:para:c001", 7, 1, "Điều 9. Các hành vi bị nghiêm cấm\n\n1. Không có giấy phép lái xe.", "luat"),
            _make_chunk("luat-p1:p007:para:c002", 7, 2, "2. Điều khiển phương tiện mà trong máu hoặc hơi thở có nồng độ cồn.", "luat"),
            _make_chunk("nd-p1:p016:para:c000", 16, 0, "Phạt tiền từ 18.000.000 đồng đến 20.000.000 đồng khi vượt đèn đỏ.", "nghi_dinh"),
        ]

    def test_bm25_and_filter_isolation(self) -> None:
        bm25 = Bm25Retriever(self.chunks)
        res_all = bm25.search("nồng độ cồn", top_k=3)
        self.assertEqual(res_all[0]["id"], "luat-p1:p007:para:c002")
        res_nd = bm25.search("nồng độ cồn", filters={"document_type": "nghi_dinh"}, top_k=3)
        self.assertEqual(res_nd, [])
        with self.assertRaises(ValueError):
            validate_filters({"invalid_field": "x"})

    def test_bm25_contextual_heading_enrichment(self) -> None:
        idx = NeighborIndex(self.chunks)
        bm25_ctx = Bm25Retriever(self.chunks, neighbor_index=idx)
        res = bm25_ctx.search("Điều 9 hành vi bị nghiêm cấm nồng độ cồn", top_k=3)
        self.assertEqual(res[0]["id"], "luat-p1:p007:para:c002")

    def test_hybrid_rrf_fuses_ranks(self) -> None:
        bm25 = Bm25Retriever(self.chunks)
        tfidf = TfidfRetriever(self.chunks)
        t_hits = tfidf.search("phạt tiền vượt đèn đỏ", top_k=3)
        self.assertEqual(t_hits[0]["id"], "nd-p1:p016:para:c000")

        dense_stub = _StubDenseRetriever([
            RetrievedChunk(**{**self.chunks[1], "score": 0.9, "strategy": "dense", "rank": 1}),
            RetrievedChunk(**{**self.chunks[0], "score": 0.7, "strategy": "dense", "rank": 2}),
        ])
        dense_hits = dense_stub.search("nồng độ cồn", top_k=3)
        lex_hits = bm25.search("nồng độ cồn", top_k=3)
        hybrid = HybridRrfRetriever(dense=dense_stub, lexical=bm25)  # type: ignore[arg-type]
        fused = hybrid.fuse(dense_hits, lex_hits, top_k=2)
        self.assertEqual(fused[0]["id"], "luat-p1:p007:para:c002")

    def test_neighbor_heading_and_context_packing_consistency(self) -> None:
        idx = NeighborIndex(self.chunks)
        found = idx.find_preceding_article_heading("luat-p1:p007:para:c002")
        self.assertIsNotNone(found)
        assert found is not None
        self.assertEqual(found[0], "luat-p1:p007:para:c001")
        self.assertIn("Điều 9", found[1])

        hit_c2 = RetrievedChunk(**{**self.chunks[1], "score": 0.85, "strategy": "bm25", "rank": 1})
        ctx, ids = pack_context_priority([hit_c2], budget=1000, neighbor_index=idx, include_heading_hint=True)
        self.assertIn("luat-p1:p007:para:c001", ids)
        self.assertIn("luat-p1:p007:para:c002", ids)
        self.assertIn("Điều 9", ctx)

        fifo_ctx, fifo_ids = pack_context_fifo([hit_c2], budget=20)
        self.assertEqual((fifo_ctx, fifo_ids), ("", []))

        # Verify pinned_ids prevent eviction across multi-turn packing.
        low_rank_pinned = RetrievedChunk(**{**self.chunks[0], "score": 0.1, "strategy": "bm25", "rank": 3})
        high_rank_new = RetrievedChunk(**{**self.chunks[2], "score": 0.9, "strategy": "bm25", "rank": 1})
        _, sticky_ids = pack_context_priority(
            [low_rank_pinned, high_rank_new],
            budget=250,
            pinned_ids=["luat-p1:p007:para:c001"],
        )
        self.assertIn("luat-p1:p007:para:c001", sticky_ids)

    def test_reranker_preserves_valid_ids_only(self) -> None:
        cands = [
            RetrievedChunk(**{**self.chunks[0], "score": 0.5, "strategy": "hybrid", "rank": 1}),
            RetrievedChunk(**{**self.chunks[1], "score": 0.4, "strategy": "hybrid", "rank": 2}),
        ]
        chat_stub = _StubChatClient(json.dumps({"ranked_ids": ["luat-p1:p007:para:c002", "invented-id"]}))
        reranker = LlmRerankRetriever(base=_StubDenseRetriever(cands), chat_client=chat_stub)  # type: ignore[arg-type]
        ranked = reranker.rerank_candidates("nồng độ cồn", cands, top_k=2)
        self.assertEqual([r["id"] for r in ranked], ["luat-p1:p007:para:c002", "luat-p1:p007:para:c001"])

    def test_safe_filter_inference(self) -> None:
        self.assertEqual(
            infer_safe_document_type_filter("Theo Luật Trật tự, an toàn giao thông đường bộ, tuổi lái xe?"),
            {"document_type": "luat"},
        )
        self.assertEqual(
            infer_safe_document_type_filter("Theo Nghị định xử phạt vi phạm hành chính..."),
            {"document_type": "nghi_dinh"},
        )
        self.assertIsNone(infer_safe_document_type_filter("Xe máy vượt đèn đỏ phạt bao nhiêu?"))

    def test_extract_json_object_handles_fences(self) -> None:
        raw = '```json\n{"action": "answer", "answer": "7 ngày [c1]", "citations": ["c1"]}\n```'
        parsed = extract_json_object(raw)
        self.assertEqual(parsed["action"], "answer")
        self.assertEqual(parsed["citations"], ["c1"])
        with self.assertRaises(ValueError):
            extract_json_object("   ")

    def test_extract_usage_fields_handles_openrouter_and_xai_ticks(self) -> None:
        or_usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15, model_extra={"cost": 0.0025})
        self.assertEqual(_extract_usage_fields(or_usage), (10, 5, 15, 0.0025))

        xai_usage = SimpleNamespace(
            prompt_tokens=305,
            completion_tokens=24,
            total_tokens=329,
            model_extra={"cost_in_usd_ticks": 3068500},
        )
        p_tok, c_tok, t_tok, cost = _extract_usage_fields(xai_usage)
        self.assertEqual((p_tok, c_tok, t_tok), (305, 24, 329))
        self.assertAlmostEqual(cost, 0.00030685, places=10)


if __name__ == "__main__":
    unittest.main()

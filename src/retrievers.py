"""Retrieval methods for Day 7: TF-IDF, BM25, pgvector Dense, Hybrid RRF, LLM Rerank, and Neighbor lookup."""
from collections import Counter
import json
import math
from pathlib import Path
import re
from typing import Protocol
import unicodedata
import psycopg

from .config import (
    ALLOWED_FILTER_FIELDS,
    BM25_B,
    BM25_K1,
    DEFAULT_CANDIDATE_K,
    RERANK_CANDIDATE_K,
    RRF_K,
)
from .providers import OpenRouterChatClient, OpenRouterEmbedder, extract_json_object
from .types import ChunkRecord, FilterMap, RetrievedChunk

_ARTICLE_HEADING_RE = re.compile(r"(?m)^(Điều\s+\d+[^\n]*)")


class RetrieverProtocol(Protocol):
    """Protocol compatible with starter.agentic.Retriever."""

    def search(
        self,
        query: str,
        *,
        filters: FilterMap | None = None,
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        ...


def tokenize_vi(text: str) -> list[str]:
    """Normalize Vietnamese text to NFC and tokenize word/number tokens."""
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


def load_chunks(path: Path) -> list[ChunkRecord]:
    """Load and validate Day 7 chunk JSON array."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Chunks file must be a non-empty JSON array: {path}")
    chunks: list[ChunkRecord] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Chunk entry must be a JSON object")
        cid = str(item["id"])
        if not cid or cid in seen:
            raise ValueError(f"Invalid or duplicate chunk id: '{cid}'")
        seen.add(cid)
        meta = item.get("metadata")
        if not isinstance(meta, dict):
            raise ValueError(f"Chunk '{cid}' metadata must be an object")
        chunks.append(
            ChunkRecord(
                id=cid,
                source_id=str(item["source_id"]),
                source=str(item["source"]),
                page=int(item["page"]),
                text=str(item["text"]),
                metadata={str(k): v for k, v in meta.items()},
            )
        )
    return chunks


def validate_filters(filters: FilterMap | None) -> FilterMap | None:
    """Validate optional filter dictionary against ALLOWED_FILTER_FIELDS."""
    if not filters:
        return None
    cleaned: FilterMap = {}
    for key, val in filters.items():
        if key not in ALLOWED_FILTER_FIELDS:
            raise ValueError(
                f"Disallowed filter field '{key}'. Permitted: {sorted(ALLOWED_FILTER_FIELDS)}"
            )
        if val is None or (isinstance(val, str) and not val.strip()):
            continue
        cleaned[key] = val
    return cleaned or None


def infer_safe_document_type_filter(question: str) -> FilterMap | None:
    """Infer document_type filter strictly from explicit legal document mentions in query text."""
    q_low = question.casefold()
    mentions_luat = (
        "theo luật trật tự" in q_low
        or "luật trật tự, an toàn giao thông" in q_low
        or "trong luật trật tự" in q_low
    )
    mentions_nghi_dinh = (
        "theo nghị định" in q_low
        or "nghị định 168" in q_low
        or "nghị định xử phạt" in q_low
    )
    if mentions_luat and not mentions_nghi_dinh:
        return {"document_type": "luat"}
    if mentions_nghi_dinh and not mentions_luat:
        return {"document_type": "nghi_dinh"}
    return None


def matches_filter(chunk: ChunkRecord, filters: FilterMap | None) -> bool:
    """Return True if chunk satisfies all exact equality metadata filters."""
    if not filters:
        return True
    for key, expected in filters.items():
        if key == "document_id":
            if chunk["source_id"] != str(expected):
                return False
        else:
            actual = chunk["metadata"].get(key)
            if actual is None or str(actual) != str(expected):
                return False
    return True


def _to_retrieved(chunk: ChunkRecord, score: float, strategy: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk["id"],
        source_id=chunk["source_id"],
        source=chunk["source"],
        page=chunk["page"],
        text=chunk["text"],
        metadata=chunk["metadata"],
        score=float(score),
        strategy=strategy,
        rank=rank,
    )


def _to_vector_literal(vector: list[float]) -> str:
    parts: list[str] = []
    for val in vector:
        num = float(val)
        if not math.isfinite(num):
            raise ValueError("Non-finite float in query vector")
        parts.append(str(num))
    return f"[{','.join(parts)}]"


class NeighborIndex:
    """Provides ordered chunk navigation and nearest article heading lookup per document."""

    def __init__(self, chunks: list[ChunkRecord]) -> None:
        self._by_id: dict[str, ChunkRecord] = {}
        self._doc_lists: dict[str, list[ChunkRecord]] = {}
        self._pos_by_id: dict[str, tuple[str, int]] = {}

        grouped: dict[str, list[ChunkRecord]] = {}
        for c in chunks:
            self._by_id[c["id"]] = c
            grouped.setdefault(c["source_id"], []).append(c)

        for sid, items in grouped.items():
            ordered = sorted(
                items,
                key=lambda x: (
                    int(x["page"]),
                    int(x["metadata"].get("chunk_index", 0) or 0),
                    x["id"],
                ),
            )
            self._doc_lists[sid] = ordered
            for idx, item in enumerate(ordered):
                self._pos_by_id[item["id"]] = (sid, idx)

    def get_chunk(self, chunk_id: str) -> ChunkRecord | None:
        return self._by_id.get(chunk_id)

    def get_neighbors(self, chunk_id: str, direction: str = "prev") -> list[ChunkRecord]:
        """Return adjacent chunks in reading order within the same source document."""
        loc = self._pos_by_id.get(chunk_id)
        if loc is None:
            return []
        sid, idx = loc
        doc_chunks = self._doc_lists[sid]
        out: list[ChunkRecord] = []
        if direction in ("prev", "both") and idx > 0:
            out.append(doc_chunks[idx - 1])
        if direction in ("next", "both") and idx + 1 < len(doc_chunks):
            out.append(doc_chunks[idx + 1])
        return out

    def find_preceding_article_heading(self, chunk_id: str, max_lookback: int = 60) -> tuple[str, str] | None:
        """If chunk begins mid-article, look back up to max_lookback chunks for nearest 'Điều ...' heading."""
        chunk = self._by_id.get(chunk_id)
        loc = self._pos_by_id.get(chunk_id)
        if chunk is None or loc is None:
            return None
        first_line = chunk["text"].lstrip().splitlines()[0] if chunk["text"].strip() else ""
        if first_line.startswith(("Điều ", "Chương ", "Mục ")):
            return None

        sid, idx = loc
        doc_chunks = self._doc_lists[sid]
        for step in range(1, max_lookback + 1):
            if idx - step < 0:
                break
            prev_c = doc_chunks[idx - step]
            matches = _ARTICLE_HEADING_RE.findall(prev_c["text"])
            if matches:
                return (prev_c["id"], matches[-1].strip())
        return None


class TfidfRetriever:
    """In-memory TF-IDF retriever matching starter/baseline.py weighting with optional filters."""

    strategy_name = "tfidf"

    def __init__(self, chunks: list[ChunkRecord]) -> None:
        self.chunks = chunks
        counts = [Counter(tokenize_vi(c["text"])) for c in chunks]
        df = Counter(term for row in counts for term in row)
        n_docs = len(chunks)
        self.idf = {term: math.log((1 + n_docs) / (1 + count)) + 1.0 for term, count in df.items()}
        self.vectors = [self._vector(row) for row in counts]

    def _vector(self, counts: Counter[str]) -> dict[str, float]:
        vec = {term: count * self.idf[term] for term, count in counts.items() if term in self.idf}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {term: v / norm for term, v in vec.items()}

    def search(
        self,
        query: str,
        *,
        filters: FilterMap | None = None,
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("Search query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        active_filters = validate_filters(filters)
        q_vec = self._vector(Counter(tokenize_vi(query)))
        scored: list[tuple[float, ChunkRecord]] = []
        for chunk, doc_vec in zip(self.chunks, self.vectors):
            if not matches_filter(chunk, active_filters):
                continue
            score = sum(val * doc_vec.get(term, 0.0) for term, val in q_vec.items())
            if score > 0.0:
                scored.append((score, chunk))
        scored.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
        return [
            _to_retrieved(chunk, score, self.strategy_name, idx)
            for idx, (score, chunk) in enumerate(scored[:top_k], start=1)
        ]


class Bm25Retriever:
    """Okapi BM25 lexical retriever over normalized Vietnamese tokens with optional metadata filter."""

    strategy_name = "bm25"

    def __init__(
        self,
        chunks: list[ChunkRecord],
        k1: float = BM25_K1,
        b: float = BM25_B,
        neighbor_index: NeighborIndex | None = None,
    ) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tfs: list[Counter[str]] = []
        self.doc_lens: list[int] = []
        df: Counter[str] = Counter()

        for chunk in chunks:
            indexed_text = chunk["text"]
            if neighbor_index is not None:
                found = neighbor_index.find_preceding_article_heading(chunk["id"])
                if found is not None:
                    indexed_text = f"{found[1]}\n{indexed_text}"
            tokens = tokenize_vi(indexed_text)
            tf = Counter(tokens)
            self.doc_tfs.append(tf)
            self.doc_lens.append(len(tokens))
            for term in tf:
                df[term] += 1

        n_docs = len(chunks)
        self.avg_len = (sum(self.doc_lens) / n_docs) if n_docs else 1.0
        self.idf = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(
        self,
        query: str,
        *,
        filters: FilterMap | None = None,
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("Search query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        active_filters = validate_filters(filters)
        q_terms = tokenize_vi(query)
        scored: list[tuple[float, ChunkRecord]] = []

        for chunk, tf, doc_len in zip(self.chunks, self.doc_tfs, self.doc_lens):
            if not matches_filter(chunk, active_filters):
                continue
            score = 0.0
            norm_len = 1.0 - self.b + self.b * (doc_len / self.avg_len)
            for term in q_terms:
                if term not in tf or term not in self.idf:
                    continue
                freq = tf[term]
                denom = freq + self.k1 * norm_len
                score += self.idf[term] * (freq * (self.k1 + 1.0)) / denom
            if score > 0.0:
                scored.append((score, chunk))

        scored.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
        return [
            _to_retrieved(chunk, score, self.strategy_name, idx)
            for idx, (score, chunk) in enumerate(scored[:top_k], start=1)
        ]


class PgVectorDenseRetriever:
    """Dense retriever querying Day 6 PostgreSQL pgvector rag_chunks table."""

    strategy_name = "dense_pgvector"

    def __init__(
        self,
        database_url: str,
        strategy: str,
        embedder: OpenRouterEmbedder,
    ) -> None:
        if not database_url.strip():
            raise ValueError("database_url must not be empty")
        if not strategy.strip():
            raise ValueError("strategy must not be empty")
        self.database_url = database_url
        self.strategy = strategy
        self.embedder = embedder

    def _build_where(self, filters: FilterMap | None) -> tuple[str, list[object]]:
        clauses = ["strategy = %s"]
        params: list[object] = [self.strategy]
        active = validate_filters(filters)
        if active:
            for key in sorted(active.keys()):
                val = active[key]
                if key == "document_id":
                    clauses.append("document_id = %s")
                    params.append(str(val))
                else:
                    clauses.append("metadata ->> %s = %s")
                    params.append(key)
                    params.append(str(val))
        return " AND ".join(clauses), params

    def search_by_vector(
        self,
        query_vector: list[float],
        *,
        filters: FilterMap | None = None,
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        vec_literal = _to_vector_literal(query_vector)
        where_sql, filter_params = self._build_where(filters)
        sql = f"""
            SELECT chunk_id, document_id, source, page, text, metadata,
                   (embedding <=> %s::vector) AS cosine_distance
            FROM rag_chunks
            WHERE {where_sql}
            ORDER BY embedding <=> %s::vector ASC, chunk_id ASC
            LIMIT %s;
        """
        params: list[object] = [vec_literal, *filter_params, vec_literal, int(top_k)]
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        results: list[RetrievedChunk] = []
        for idx, row in enumerate(rows, start=1):
            raw_meta = row[5]
            if isinstance(raw_meta, dict):
                meta = {str(k): v for k, v in raw_meta.items()}
            elif isinstance(raw_meta, str):
                meta = {str(k): v for k, v in json.loads(raw_meta).items()}
            else:
                raise TypeError(f"Unexpected metadata type: {type(raw_meta).__name__}")
            dist = float(row[6])
            results.append(
                RetrievedChunk(
                    id=str(row[0]),
                    source_id=str(row[1]),
                    source=str(row[2]),
                    page=int(row[3]),
                    text=str(row[4]),
                    metadata=meta,
                    score=1.0 - dist,
                    strategy=self.strategy_name,
                    rank=idx,
                )
            )
        return results

    def search(
        self,
        query: str,
        *,
        filters: FilterMap | None = None,
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("Search query must not be empty")
        q_vec = self.embedder.embed_query(query)
        return self.search_by_vector(q_vec, filters=filters, top_k=top_k)


class HybridRrfRetriever:
    """Combines pgvector dense retrieval and BM25 lexical retrieval via Reciprocal Rank Fusion."""

    strategy_name = "hybrid_rrf"

    def __init__(
        self,
        dense: PgVectorDenseRetriever,
        lexical: Bm25Retriever,
        rrf_k: int = RRF_K,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        dense_weight: float = 1.0,
        lexical_weight: float = 1.0,
    ) -> None:
        self.dense = dense
        self.lexical = lexical
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight

    def fuse(
        self,
        dense_hits: list[RetrievedChunk],
        lexical_hits: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Fuse pre-computed dense and lexical rankings using weighted RRF."""
        scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievedChunk] = {}

        for rank_idx, item in enumerate(dense_hits, start=1):
            cid = item["id"]
            chunk_map[cid] = item
            scores[cid] = scores.get(cid, 0.0) + self.dense_weight / (self.rrf_k + rank_idx)

        for rank_idx, item in enumerate(lexical_hits, start=1):
            cid = item["id"]
            if cid not in chunk_map:
                chunk_map[cid] = item
            scores[cid] = scores.get(cid, 0.0) + self.lexical_weight / (self.rrf_k + rank_idx)

        ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        out: list[RetrievedChunk] = []
        for idx, (cid, rrf_score) in enumerate(ordered[:top_k], start=1):
            base = chunk_map[cid]
            out.append(
                RetrievedChunk(
                    id=base["id"],
                    source_id=base["source_id"],
                    source=base["source"],
                    page=base["page"],
                    text=base["text"],
                    metadata=base["metadata"],
                    score=float(rrf_score),
                    strategy=self.strategy_name,
                    rank=idx,
                )
            )
        return out

    def search_with_vector(
        self,
        query: str,
        query_vector: list[float],
        *,
        filters: FilterMap | None = None,
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        fetch_k = max(self.candidate_k, top_k)
        dense_hits = self.dense.search_by_vector(query_vector, filters=filters, top_k=fetch_k)
        lexical_hits = self.lexical.search(query, filters=filters, top_k=fetch_k)
        return self.fuse(dense_hits, lexical_hits, top_k=top_k)

    def search(
        self,
        query: str,
        *,
        filters: FilterMap | None = None,
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("Search query must not be empty")
        fetch_k = max(self.candidate_k, top_k)
        dense_hits = self.dense.search(query, filters=filters, top_k=fetch_k)
        lexical_hits = self.lexical.search(query, filters=filters, top_k=fetch_k)
        return self.fuse(dense_hits, lexical_hits, top_k=top_k)


class LlmRerankRetriever:
    """Two-stage retrieval: fetches candidate_k from base retriever and reranks via LLM."""

    strategy_name = "hybrid_rerank"

    def __init__(
        self,
        base: RetrieverProtocol,
        chat_client: OpenRouterChatClient,
        candidate_k: int = RERANK_CANDIDATE_K,
    ) -> None:
        self.base = base
        self.chat_client = chat_client
        self.candidate_k = candidate_k

    def rerank_candidates(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Rerank already-fetched candidates using OpenRouter chat model in JSON mode."""
        if not candidates:
            return []
        if len(candidates) == 1:
            return [
                RetrievedChunk(
                    **{**candidates[0], "score": 1.0, "strategy": self.strategy_name, "rank": 1}
                )
            ]

        cand_lines: list[str] = []
        valid_ids: set[str] = set()
        cand_map: dict[str, RetrievedChunk] = {}
        for idx, c in enumerate(candidates, start=1):
            valid_ids.add(c["id"])
            cand_map[c["id"]] = c
            snippet = c["text"].replace("\n", " ").strip()[:550]
            cand_lines.append(f"{idx}. ID={c['id']} | page={c['page']} | text={snippet}")

        system_prompt = (
            "Bạn là bộ xếp hạng đoạn văn pháp lý giao thông Việt Nam. "
            "Nhiệm vụ: xếp hạng các đoạn ứng viên theo mức độ liên quan trực tiếp để trả lời câu hỏi. "
            "Lưu ý phân biệt đúng loại phương tiện (xe ô tô vs xe mô tô/xe máy vs xe thô sơ), "
            "đúng loại văn bản (Luật vs Nghị định mức phạt), và mốc thời gian hiệu lực nếu có. "
            "Chỉ trả về JSON object dạng: {\"ranked_ids\": [\"id_tot_nhat\", ...]} gồm đúng các ID có trong danh sách."
        )
        user_prompt = (
            f"Câu hỏi: {query}\n\n"
            f"Danh sách ứng viên:\n" + "\n".join(cand_lines) + "\n\n"
            "Hãy trả về đúng 1 JSON {\"ranked_ids\": [...]} xếp tối đa 5 ID liên quan nhất từ cao xuống thấp, không giải thích thêm."
        )
        res = self.chat_client.complete(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=3000,
            json_mode=False,
        )
        parsed = extract_json_object(res.content)
        raw_ids = parsed.get("ranked_ids")
        if not isinstance(raw_ids, list):
            raise ValueError(f"Reranker output missing ranked_ids list: {res.content}")

        ordered_ids: list[str] = []
        seen: set[str] = set()
        for item in raw_ids:
            sid = str(item)
            if sid in valid_ids and sid not in seen:
                ordered_ids.append(sid)
                seen.add(sid)
        # Preserve original order for any valid candidate omitted by the model.
        for c in candidates:
            if c["id"] not in seen:
                ordered_ids.append(c["id"])
                seen.add(c["id"])

        out: list[RetrievedChunk] = []
        for idx, cid in enumerate(ordered_ids[:top_k], start=1):
            base_c = cand_map[cid]
            out.append(
                RetrievedChunk(
                    id=base_c["id"],
                    source_id=base_c["source_id"],
                    source=base_c["source"],
                    page=base_c["page"],
                    text=base_c["text"],
                    metadata=base_c["metadata"],
                    score=1.0 / idx,
                    strategy=self.strategy_name,
                    rank=idx,
                )
            )
        return out

    def search(
        self,
        query: str,
        *,
        filters: FilterMap | None = None,
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        fetch_k = max(self.candidate_k, top_k)
        candidates = self.base.search(query, filters=filters, top_k=fetch_k)
        return self.rerank_candidates(query, candidates, top_k=top_k)

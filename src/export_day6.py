"""Export Day 6 pgvector chunks to Day 7 JSON schema and verify vector store health."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import psycopg

from .config import CHUNKS_PATH, load_settings


@dataclass(frozen=True)
class StoreHealth:
    """Verified PostgreSQL + pgvector metadata for Day 6 strategy."""

    database: str
    postgres_version: str
    pgvector_version: str
    strategy: str
    chunk_count: int
    embedding_dimensions: int


def _decode_metadata(raw_value: object) -> dict[str, object]:
    if isinstance(raw_value, dict):
        return {str(k): v for k, v in raw_value.items()}
    if isinstance(raw_value, str):
        parsed = json.loads(raw_value)
        if isinstance(parsed, dict):
            return {str(k): v for k, v in parsed.items()}
    raise TypeError(f"Unexpected metadata JSONB type: {type(raw_value).__name__}")


def inspect_and_export_day6(
    database_url: str,
    strategy: str,
    output_path: Path,
) -> tuple[StoreHealth, list[dict[str, object]]]:
    """Query Day 6 chunks from rag_chunks, validate Day 7 contract, and write JSON."""
    if not database_url.strip():
        raise ValueError("database_url must not be empty")
    if not strategy.strip():
        raise ValueError("strategy must not be empty")

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT current_database(),
                       version(),
                       COALESCE(
                           (SELECT extversion FROM pg_extension WHERE extname = 'vector'),
                           'missing'
                       );
                """
            )
            db_row = cur.fetchone()
            if db_row is None:
                raise RuntimeError("Could not read PostgreSQL database info")

            cur.execute(
                """
                SELECT chunk_id, document_id, source, page, text, metadata, vector_dims(embedding)
                FROM rag_chunks
                WHERE strategy = %s
                ORDER BY document_id ASC, page ASC, chunk_id ASC;
                """,
                (strategy,),
            )
            rows = cur.fetchall()

    if not rows:
        raise RuntimeError(f"No rows found in rag_chunks for strategy '{strategy}'")

    dims_set = {int(r[6]) for r in rows}
    if len(dims_set) != 1:
        raise RuntimeError(f"Inconsistent vector dimensions in strategy '{strategy}': {dims_set}")
    dims = next(iter(dims_set))

    chunks: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for row in rows:
        cid = str(row[0])
        if not cid or cid in seen_ids:
            raise ValueError(f"Duplicate or empty chunk_id: '{cid}'")
        seen_ids.add(cid)
        text = str(row[4])
        if not text.strip():
            raise ValueError(f"Chunk '{cid}' has empty text")
        chunks.append(
            {
                "id": cid,
                "source_id": str(row[1]),
                "source": str(row[2]),
                "page": int(row[3]),
                "text": text,
                "metadata": _decode_metadata(row[5]),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    health = StoreHealth(
        database=str(db_row[0]),
        postgres_version=str(db_row[1]),
        pgvector_version=str(db_row[2]),
        strategy=strategy,
        chunk_count=len(chunks),
        embedding_dimensions=dims,
    )
    return health, chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Day 6 pgvector chunks to Day 7 JSON")
    parser.add_argument("--strategy", default=None, help="Override DAY6_STRATEGY from .env")
    parser.add_argument("--output", type=Path, default=CHUNKS_PATH)
    args = parser.parse_args()

    settings = load_settings()
    strategy = args.strategy or settings.day6_strategy
    health, _ = inspect_and_export_day6(settings.database_url, strategy, args.output)
    print(
        json.dumps(
            {
                "database": health.database,
                "postgres_version": health.postgres_version,
                "pgvector_version": health.pgvector_version,
                "strategy": health.strategy,
                "chunk_count": health.chunk_count,
                "embedding_dimensions": health.embedding_dimensions,
                "output_path": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Context selection and budget packing strategies for Day 7."""
from .retrievers import NeighborIndex
from .types import ChunkRecord, RetrievedChunk


def format_block(chunk: ChunkRecord | RetrievedChunk, prefix_note: str | None = None) -> str:
    """Format one chunk with traceable source header matching starter/baseline.py."""
    header = f"[{chunk['id']} | {chunk['source']} | page {chunk['page']}]"
    if prefix_note:
        return f"{header}\n{prefix_note}\n{chunk['text']}"
    return f"{header}\n{chunk['text']}"


def pack_context_fifo(
    results: list[RetrievedChunk] | list[ChunkRecord],
    budget: int = 800,
) -> tuple[str, list[str]]:
    """Pack unique chunks in arrival order without exceeding character budget."""
    blocks: list[str] = []
    ids: list[str] = []
    seen: set[str] = set()
    for c in results:
        cid = c["id"]
        if cid in seen:
            continue
        seen.add(cid)
        block = format_block(c)
        candidate = "\n\n".join(blocks + [block])
        if len(candidate) <= budget:
            blocks.append(block)
            ids.append(cid)
    return "\n\n".join(blocks), ids


def pack_context_priority(
    results: list[RetrievedChunk],
    budget: int = 2400,
    neighbor_index: NeighborIndex | None = None,
    include_heading_hint: bool = False,
    pinned_ids: list[str] | None = None,
) -> tuple[str, list[str]]:
    """Deduplicate keeping pinned_ids first, then highest-rank priority, optionally with Article heading."""
    best_by_id: dict[str, RetrievedChunk] = {}
    order_pos: dict[str, int] = {}
    for pos, item in enumerate(results):
        cid = item["id"]
        if cid not in best_by_id:
            best_by_id[cid] = item
            order_pos[cid] = pos
        else:
            prev = best_by_id[cid]
            if (item["rank"], -item["score"]) < (prev["rank"], -prev["score"]):
                best_by_id[cid] = item
                order_pos[cid] = pos

    pinned_order = {cid: idx for idx, cid in enumerate(pinned_ids or [])}
    ordered = sorted(
        best_by_id.values(),
        key=lambda c: (
            0 if c["id"] in pinned_order else 1,
            pinned_order.get(c["id"], 0),
            c["rank"],
            -c["score"],
            order_pos[c["id"]],
        ),
    )

    blocks: list[str] = []
    ids: list[str] = []
    included: set[str] = set()
    for c in ordered:
        cid = c["id"]
        if cid in included:
            continue

        # Optionally include traced Article heading block from preceding chunk before mid-article chunk.
        if include_heading_hint and neighbor_index is not None:
            found = neighbor_index.find_preceding_article_heading(cid)
            if found is not None:
                heading_cid, heading_text = found
                if heading_cid not in included:
                    h_chunk = neighbor_index.get_chunk(heading_cid)
                    if h_chunk is not None:
                        h_block = (
                            f"[{h_chunk['id']} | {h_chunk['source']} | page {h_chunk['page']}]\n"
                            f"(Đầu mục Điều liên quan): {heading_text}"
                        )
                        c_block = format_block(c)
                        if len("\n\n".join(blocks + [h_block, c_block])) <= budget:
                            blocks.append(h_block)
                            ids.append(heading_cid)
                            included.add(heading_cid)
                            blocks.append(c_block)
                            ids.append(cid)
                            included.add(cid)
                            continue

        block = format_block(c)
        if len("\n\n".join(blocks + [block])) <= budget:
            blocks.append(block)
            ids.append(cid)
            included.add(cid)

    return "\n\n".join(blocks), ids

"""Shared typed data contracts for Day 7 retrieval and agentic RAG."""
from dataclasses import dataclass, field
from typing import Literal, TypedDict

MetadataScalar = str | int | float | bool
FilterMap = dict[str, MetadataScalar]


class ChunkRecord(TypedDict):
    """Normalized Day 7 chunk contract."""

    id: str
    source_id: str
    source: str
    page: int
    text: str
    metadata: dict[str, object]


class RetrievedChunk(TypedDict):
    """Search result chunk returned by a retriever."""

    id: str
    source_id: str
    source: str
    page: int
    text: str
    metadata: dict[str, object]
    score: float
    strategy: str
    rank: int


class GoldSpan(TypedDict):
    """Single ground-truth evidence span inside a chunk."""

    source_id: str
    page: int
    evidence: str


class QuestionRecord(TypedDict, total=False):
    """Benchmark retrieval question record."""

    id: str
    split: Literal["dev", "test"]
    type: str
    question: str
    expected_answer: str
    gold: list[GoldSpan]
    filter: FilterMap
    requires_neighbor_heading: bool


class AgentCaseRecord(TypedDict, total=False):
    """Benchmark agent case record (answerable or unanswerable)."""

    id: str
    split: Literal["dev", "test"]
    type: str
    question: str
    answerable: bool
    expected_behavior: str
    absence_reason: str
    filter: FilterMap


@dataclass
class UsageAccumulator:
    """Tracks API token usage and OpenRouter reported USD cost."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    api_calls: int = 0

    def add(self, prompt_tokens: int, completion_tokens: int, total_tokens: int, cost_usd: float) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        self.cost_usd += cost_usd
        self.api_calls += 1

    def snapshot(self) -> tuple[int, int, int, float, int]:
        return (
            self.prompt_tokens,
            self.completion_tokens,
            self.total_tokens,
            self.cost_usd,
            self.api_calls,
        )

    @staticmethod
    def delta(
        before: tuple[int, int, int, float, int],
        after: tuple[int, int, int, float, int],
    ) -> dict[str, object]:
        return {
            "prompt_tokens": after[0] - before[0],
            "completion_tokens": after[1] - before[1],
            "total_tokens": after[2] - before[2],
            "cost_usd": round(after[3] - before[3], 8),
            "api_calls": after[4] - before[4],
        }


@dataclass(frozen=True)
class ChatToolCall:
    """Normalized tool call emitted by the chat model."""

    call_id: str
    name: str
    arguments_json: str


@dataclass(frozen=True)
class ChatCompletionResult:
    """Normalized chat completion response."""

    content: str
    tool_calls: list[ChatToolCall] = field(default_factory=list)
    finish_reason: str = ""
    raw_assistant_message: dict[str, object] = field(default_factory=dict)

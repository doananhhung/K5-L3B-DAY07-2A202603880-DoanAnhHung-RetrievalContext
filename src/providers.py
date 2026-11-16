"""OpenAI-compatible embedding and chat completion clients (OpenRouter, xAI, Groq) with usage tracking."""
import json
import math
import time
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from .types import ChatCompletionResult, ChatToolCall, UsageAccumulator

# xAI reports usage billing in integer ticks where 1 USD = 1e10 ticks.
XAI_COST_TICKS_PER_USD = 10_000_000_000.0


def extract_json_object(raw_text: str) -> dict[str, object]:
    """Parse a JSON object from raw model output, stripping markdown code fences if present."""
    cleaned = raw_text.strip()
    if not cleaned:
        raise ValueError("Empty model response; expected JSON object")
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return {str(k): v for k, v in parsed.items()}
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    idx = cleaned.find("{")
    if idx != -1:
        try:
            parsed_obj, _ = decoder.raw_decode(cleaned[idx:])
            if isinstance(parsed_obj, dict):
                return {str(k): v for k, v in parsed_obj.items()}
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON object in model output: {raw_text[:200]}") from exc
    raise ValueError(f"Could not extract valid JSON object from model output: {raw_text[:200]}")


def _extract_usage_fields(usage_obj: object) -> tuple[int, int, int, float]:
    """Safely extract token counts and optional provider USD cost from SDK usage object."""
    if usage_obj is None:
        return (0, 0, 0, 0.0)
    prompt_tok = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
    comp_tok = int(getattr(usage_obj, "completion_tokens", 0) or 0)
    total_tok = int(getattr(usage_obj, "total_tokens", prompt_tok + comp_tok) or (prompt_tok + comp_tok))
    raw_cost = getattr(usage_obj, "cost", None)
    extra = getattr(usage_obj, "model_extra", None)
    if raw_cost is None and isinstance(extra, dict):
        raw_cost = extra.get("cost")
    if raw_cost is not None:
        cost_usd = float(raw_cost)
    else:
        raw_ticks = getattr(usage_obj, "cost_in_usd_ticks", None)
        if raw_ticks is None and isinstance(extra, dict):
            raw_ticks = extra.get("cost_in_usd_ticks")
        cost_usd = (float(raw_ticks) / XAI_COST_TICKS_PER_USD) if raw_ticks is not None else 0.0
    return (prompt_tok, comp_tok, total_tok, cost_usd)


class OpenRouterEmbedder:
    """OpenRouter-backed query embedder compatible with Day 6 stored vectors."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        expected_dimensions: int | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("embedding model_name must not be empty")
        if not api_key.strip():
            raise ValueError("openrouter api_key must not be empty")
        self.model_name = model_name
        self.expected_dimensions = expected_dimensions
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=30.0, max_retries=0)
        self.usage = UsageAccumulator()
        self.dimensions: int = expected_dimensions or 0

    def embed_query(self, question: str) -> list[float]:
        """Embed a non-empty query string and verify dimensionality."""
        cleaned = question.strip()
        if not cleaned:
            raise ValueError("Query text must not be empty")

        last_err: Exception | None = None
        for attempt in range(3):
            try:
                resp = self.client.embeddings.create(
                    model=self.model_name,
                    input=[cleaned],
                    encoding_format="float",
                )
                p_tok, c_tok, t_tok, cost = _extract_usage_fields(resp.usage)
                self.usage.add(p_tok, c_tok, t_tok, cost)
                if not resp.data:
                    raise RuntimeError("Empty embedding response from OpenRouter")
                vec = [float(x) for x in resp.data[0].embedding]
                if not vec or any(not math.isfinite(x) for x in vec):
                    raise RuntimeError("Non-finite or empty embedding vector returned")
                if self.dimensions == 0:
                    self.dimensions = len(vec)
                elif len(vec) != self.dimensions:
                    raise RuntimeError(
                        f"Embedding dimension mismatch: expected {self.dimensions}, got {len(vec)}"
                    )
                return vec
            except (APIConnectionError, APITimeoutError) as exc:
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
            except APIStatusError as exc:
                if exc.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                    last_err = exc
                    time.sleep(2.0 * (attempt + 1))
                else:
                    raise RuntimeError(
                        f"OpenRouter embeddings error HTTP {exc.status_code} for {self.model_name}"
                    ) from exc

        raise RuntimeError(f"OpenRouter embeddings failed after retries: {last_err}") from last_err


class OpenRouterChatClient:
    """OpenAI-compatible chat client supporting deterministic tool calling and JSON mode."""

    def __init__(self, model_name: str, api_key: str, base_url: str, timeout: float = 90.0) -> None:
        if not model_name.strip():
            raise ValueError("chat model_name must not be empty")
        if not api_key.strip():
            raise ValueError("chat api_key must not be empty")
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)
        self.usage = UsageAccumulator()
        if model_name.startswith("openai/gpt-oss"):
            self.default_reasoning_effort: str | None = "low"
        elif model_name.startswith("grok-4.3"):
            # Use non-reasoning mode by default on grok-4.3 for low-latency tool/JSON turns.
            self.default_reasoning_effort = "none"
        else:
            self.default_reasoning_effort = None

    def complete(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> ChatCompletionResult:
        """Send a chat completion request with retries on transient transport/rate errors."""
        last_err: Exception | None = None
        for attempt in range(6):
            try:
                if tools and json_mode:
                    if self.default_reasoning_effort:
                        resp = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,  # type: ignore[arg-type]
                            temperature=temperature,
                            max_tokens=max_tokens,
                            tools=tools,  # type: ignore[arg-type]
                            tool_choice="auto",
                            response_format={"type": "json_object"},
                            reasoning_effort=self.default_reasoning_effort,  # type: ignore[call-overload]
                        )
                    else:
                        resp = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,  # type: ignore[arg-type]
                            temperature=temperature,
                            max_tokens=max_tokens,
                            tools=tools,  # type: ignore[arg-type]
                            tool_choice="auto",
                            response_format={"type": "json_object"},
                        )
                elif tools:
                    if self.default_reasoning_effort:
                        resp = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,  # type: ignore[arg-type]
                            temperature=temperature,
                            max_tokens=max_tokens,
                            tools=tools,  # type: ignore[arg-type]
                            tool_choice="auto",
                            reasoning_effort=self.default_reasoning_effort,  # type: ignore[call-overload]
                        )
                    else:
                        resp = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,  # type: ignore[arg-type]
                            temperature=temperature,
                            max_tokens=max_tokens,
                            tools=tools,  # type: ignore[arg-type]
                            tool_choice="auto",
                        )
                elif json_mode:
                    if self.default_reasoning_effort:
                        resp = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,  # type: ignore[arg-type]
                            temperature=temperature,
                            max_tokens=max_tokens,
                            response_format={"type": "json_object"},
                            reasoning_effort=self.default_reasoning_effort,  # type: ignore[call-overload]
                        )
                    else:
                        resp = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,  # type: ignore[arg-type]
                            temperature=temperature,
                            max_tokens=max_tokens,
                            response_format={"type": "json_object"},
                        )
                else:
                    if self.default_reasoning_effort:
                        resp = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,  # type: ignore[arg-type]
                            temperature=temperature,
                            max_tokens=max_tokens,
                            reasoning_effort=self.default_reasoning_effort,  # type: ignore[call-overload]
                        )
                    else:
                        resp = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,  # type: ignore[arg-type]
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                p_tok, c_tok, t_tok, cost = _extract_usage_fields(resp.usage)
                self.usage.add(p_tok, c_tok, t_tok, cost)

                if not resp.choices:
                    raise RuntimeError("Chat provider returned zero choices")
                choice = resp.choices[0]
                msg = choice.message
                content = str(msg.content or "")

                parsed_calls: list[ChatToolCall] = []
                raw_tool_calls: list[dict[str, object]] = []
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        func = getattr(tc, "function", None)
                        if func is None:
                            continue
                        parsed_calls.append(
                            ChatToolCall(
                                call_id=str(tc.id),
                                name=str(func.name),
                                arguments_json=str(func.arguments or "{}"),
                            )
                        )
                        raw_tool_calls.append(
                            {
                                "id": str(tc.id),
                                "type": "function",
                                "function": {
                                    "name": str(func.name),
                                    "arguments": str(func.arguments or "{}"),
                                },
                            }
                        )

                raw_assistant: dict[str, object] = {
                    "role": "assistant",
                    "content": msg.content if msg.content is not None else "",
                }
                if raw_tool_calls:
                    raw_assistant["tool_calls"] = raw_tool_calls

                return ChatCompletionResult(
                    content=content,
                    tool_calls=parsed_calls,
                    finish_reason=str(choice.finish_reason or ""),
                    raw_assistant_message=raw_assistant,
                )
            except (APIConnectionError, APITimeoutError) as exc:
                last_err = exc
                time.sleep(2.0 * (attempt + 1))
            except APIStatusError as exc:
                err_text = str(exc.body or exc)
                is_daily_or_quota = (
                    exc.status_code == 402
                    or "free-models-per-day" in err_text
                    or "tokens per day" in err_text.lower()
                )
                if exc.status_code in {429, 500, 502, 503, 504} and not is_daily_or_quota and attempt < 5:
                    last_err = exc
                    retry_hdr = exc.response.headers.get("retry-after") if exc.response is not None else None
                    wait_s = min(35.0, 2.0 * (2**attempt))
                    if retry_hdr is not None:
                        try:
                            wait_s = min(40.0, max(2.0, float(retry_hdr) + 1.0))
                        except ValueError:
                            pass
                    time.sleep(wait_s)
                else:
                    raise RuntimeError(
                        f"Chat API error HTTP {exc.status_code} for {self.model_name}: {err_text}"
                    ) from exc

        raise RuntimeError(f"Chat completion failed after retries: {last_err}") from last_err

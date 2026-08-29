"""OneTwo-compatible Anthropic Claude backend.

Wraps Anthropic's Messages API so that the DomainExpertAgent's existing
OneTwo ReAct flow can drive Claude models without forking OneTwo or
introducing a parallel agent loop.

The surface mirrors `onetwo.backends.openai_api.OpenAIAPI` closely enough
that `onetwo.agents.react.ReActAgent` cannot distinguish the two. The key
ReAct call — ``generate_text(stop=stop_sequences)`` — works unchanged
because Anthropic's Messages API supports ``stop_sequences`` natively.

Activation is gated by both the ``anthropic`` SDK and ``onetwo`` being
importable. Either missing causes ``AnthropicAPI`` to refuse construction
with a clear error.
"""

import dataclasses
import os
from collections.abc import Iterable, Sequence
from typing import Any, List, Mapping, Optional

try:
    import anthropic
    _ANTHROPIC_SDK_AVAILABLE = True
except ImportError:
    anthropic = None  # type: ignore[assignment]
    _ANTHROPIC_SDK_AVAILABLE = False

try:
    from onetwo.builtins import llm
    from onetwo.builtins import formatting
    from onetwo.core.content import ChunkList
    _ONETWO_AVAILABLE = True
except ImportError:
    llm = None  # type: ignore[assignment]
    formatting = None  # type: ignore[assignment]
    ChunkList = None  # type: ignore[assignment]
    _ONETWO_AVAILABLE = False


def is_available() -> bool:
    """Return True when both the anthropic SDK and onetwo are importable."""
    return _ANTHROPIC_SDK_AVAILABLE and _ONETWO_AVAILABLE


@dataclasses.dataclass
class AnthropicAPI:
    """OneTwo backend wrapper for Anthropic Claude models.

    Mirrors the public shape of ``onetwo.backends.openai_api.OpenAIAPI`` —
    a dataclass with ``api_key``, ``model_name``, generation hyper-params,
    and a ``register()`` method that wires the instance into OneTwo's
    ``llm.generate_text`` / ``llm.chat`` builtins.

    Usage::

        backend = AnthropicAPI(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model_name="claude-sonnet-4-5-20250929",
            temperature=0.6,
        )
        backend.register()
        # ReAct and other OneTwo agents now route through Claude.
    """

    api_key: Optional[str] = None
    model_name: str = "claude-sonnet-4-5-20250929"
    temperature: Optional[float] = None
    # Anthropic's Messages API requires max_tokens. 4096 is generous for
    # Domain Expert ReAct steps (which are short Thought/Act/Observe text).
    max_tokens: int = 4096
    # Reserved for parity with OpenAIAPI; not used in our request.
    batch_size: int = 1

    def __post_init__(self) -> None:
        if not _ANTHROPIC_SDK_AVAILABLE:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )
        if not _ONETWO_AVAILABLE:
            raise ImportError(
                "onetwo package required to register the Anthropic backend."
            )
        api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key required. Pass api_key or set "
                "ANTHROPIC_API_KEY environment variable."
            )
        self.api_key = api_key
        self._client = anthropic.Anthropic(api_key=self.api_key)
        # Match OpenAIAPI's counters attribute so OneTwo's instrumentation
        # path (which sometimes pokes at `_counters['generate_text'] += 1`)
        # doesn't AttributeError if it's invoked.
        self._counters: dict[str, int] = {}

    # ------------------------------------------------------------------
    # OneTwo registration
    # ------------------------------------------------------------------

    def register(self, name: Optional[str] = None) -> None:
        """Register this backend with OneTwo's llm builtin registry.

        ReAct only routes through ``llm.generate_text``, so that's all we
        configure. ``llm.chat`` / ``llm.instruct`` / ``llm.generate_object``
        are intentionally left at their reset defaults — any OneTwo path
        that tries to use them with this backend will raise a clear
        "default not configured" error rather than silently misbehave.
        """
        del name
        llm.reset_defaults()
        llm.generate_text.configure(
            self.generate_text,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=None,
        )

    # ------------------------------------------------------------------
    # generate_text — the only call ReAct's prompt template makes
    # ------------------------------------------------------------------

    def generate_text(
        self,
        prompt: str | ChunkList,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        decoding_constraint: str | None = None,
        include_details: bool = False,
    ):
        """See builtins.llm.generate_text.

        Honors the OneTwo generate_text signature so that
        ``generate_text(stop=stop_sequences)`` from ReAct works unchanged.
        Anthropic does not implement decoding_constraint; we silently
        ignore it (consistent with OpenAIAPI's ``del decoding_constraint``).
        """
        del decoding_constraint
        self._counters["generate_text"] = self._counters.get("generate_text", 0) + 1

        prompt_text = _coerce_prompt_to_text(prompt)
        messages = [{"role": "user", "content": prompt_text}]

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }
        effective_temp = temperature if temperature is not None else self.temperature
        if effective_temp is not None:
            kwargs["temperature"] = effective_temp
        if stop:
            # Anthropic accepts up to 4 stop sequences via stop_sequences (plural).
            kwargs["stop_sequences"] = list(stop)[:4]
        if top_p is not None:
            kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = top_k

        response = self._client.messages.create(**kwargs)
        text = _extract_text(response)

        if include_details:
            details: Mapping[str, Any] = {
                "model": getattr(response, "model", self.model_name),
                "stop_reason": getattr(response, "stop_reason", None),
                "usage": {
                    "input_tokens": getattr(response.usage, "input_tokens", None)
                    if getattr(response, "usage", None)
                    else None,
                    "output_tokens": getattr(response.usage, "output_tokens", None)
                    if getattr(response, "usage", None)
                    else None,
                },
            }
            return text, details
        return text

    # ------------------------------------------------------------------
    # chat — minimal compatibility shim
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: Iterable[Any],
        *,
        formatter: Any = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        **_unused_kwargs: Any,
    ) -> str:
        """Minimal chat surface. ReAct doesn't call this directly, but
        OneTwo's ``llm.chat`` builtin might be invoked by other paths.

        Translates OneTwo's role-tagged messages to Anthropic's
        ``user``/``assistant`` roles, and lifts the first ``system``-role
        message to the top-level ``system`` parameter (Anthropic's API
        treats system as a separate field, not a role inside ``messages``).
        """
        del formatter, _unused_kwargs
        self._counters["chat"] = self._counters.get("chat", 0) + 1

        anth_messages: List[dict[str, str]] = []
        system_text: Optional[str] = None
        for msg in messages:
            role = str(getattr(msg, "role", "user")).lower()
            content = str(getattr(msg, "content", msg))
            if "system" in role:
                # First system message wins; subsequent ones are concatenated.
                system_text = (system_text + "\n" + content) if system_text else content
                continue
            anth_role = "assistant" if ("assistant" in role or "model" in role) else "user"
            anth_messages.append({"role": anth_role, "content": content})

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": anth_messages,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }
        if system_text:
            kwargs["system"] = system_text
        effective_temp = temperature if temperature is not None else self.temperature
        if effective_temp is not None:
            kwargs["temperature"] = effective_temp
        if stop:
            kwargs["stop_sequences"] = list(stop)[:4]

        response = self._client.messages.create(**kwargs)
        return _extract_text(response)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _coerce_prompt_to_text(prompt: Any) -> str:
    """Reduce a OneTwo prompt (str | ChunkList | other) to a single string.

    Anthropic's Messages API expects string content. ChunkLists carry
    multimodal content; ReAct only emits text chunks, so concatenating
    string-shaped chunks is sufficient. Anything else falls back to ``str()``.
    """
    if isinstance(prompt, str):
        return prompt
    # Best-effort handling of OneTwo's ChunkList: iterate, take string content.
    try:
        parts: List[str] = []
        for chunk in prompt:  # type: ignore[union-attr]
            content_type = getattr(chunk, "content_type", None)
            if content_type == "str":
                parts.append(str(chunk.content))
            elif content_type is None:
                parts.append(str(chunk))
            # Other content types (e.g., images) silently dropped for now.
        if parts:
            return "".join(parts)
    except TypeError:
        pass
    return str(prompt)


def _extract_text(response: Any) -> str:
    """Pull text out of an Anthropic ``Message`` response.

    The Messages API returns ``content`` as a list of typed blocks. For
    text-only models there is exactly one ``TextBlock``; we concatenate any
    that have a ``.text`` attribute to be safe.
    """
    content = getattr(response, "content", None) or []
    parts = [getattr(b, "text", "") for b in content if hasattr(b, "text")]
    return "".join(parts)

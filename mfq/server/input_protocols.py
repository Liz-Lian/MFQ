"""Architecture-registered prompt encoders at the managed API boundary."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from mfq.server.capabilities import (
    capabilities_for_architecture,
    normalize_architecture,
)
from mfq.server.deepseek_v4_prompt import render_deepseek_v4_prompt
from mfq.server.deepseek_v41_prompt import render_deepseek_v41_prompt
from mfq.server.models import ResponseFormat, ToolChoice, ToolDefinition

PromptRenderer = Callable[..., str]


@dataclass(frozen=True)
class _InputProtocolRegistration:
    families: frozenset[str]
    aliases: frozenset[str]
    prefixes: tuple[str, ...]
    renderer: PromptRenderer


_REGISTRY = (
    _InputProtocolRegistration(
        families=frozenset({"deepseek_v41"}),
        aliases=frozenset(
            {"deepseek_v41", "deepseek_v41_text", "deepseek_v41_vision"}
        ),
        prefixes=("deepseek_v41",),
        renderer=render_deepseek_v41_prompt,
    ),
    _InputProtocolRegistration(
        families=frozenset({"deepseek_v4"}),
        aliases=frozenset({"deepseek_v4", "deepseek_v4_vision"}),
        prefixes=("deepseek_v4",),
        renderer=render_deepseek_v4_prompt,
    ),
)


def render_preformatted_prompt(
    architecture: str,
    messages: Sequence[dict[str, Any]],
    *,
    tools: Sequence[ToolDefinition] = (),
    tool_choice: ToolChoice = "auto",
    response_format: ResponseFormat | None = None,
    enable_thinking: bool = True,
    reasoning_effort: str | None = None,
    parallel_tool_calls: bool = True,
) -> str | None:
    """Render processor-owned protocols; return ``None`` for Jinja models."""

    identity = normalize_architecture(architecture)
    family = capabilities_for_architecture(architecture).architecture_family
    for registration in _REGISTRY:
        if (
            family in registration.families
            or identity in registration.aliases
            or identity.startswith(registration.prefixes)
        ):
            return registration.renderer(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                enable_thinking=enable_thinking,
                reasoning_effort=reasoning_effort,
                parallel_tool_calls=parallel_tool_calls,
            )
    return None


__all__ = ["render_preformatted_prompt"]

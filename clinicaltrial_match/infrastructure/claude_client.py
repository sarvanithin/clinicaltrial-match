"""
Async Anthropic client wrapper with model routing and retry logic.

Routes to haiku (fast/cheap) for parsing/extraction tasks and
sonnet (reasoning) for complex constraint evaluation.
"""

from __future__ import annotations

import os
from typing import Any

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from clinicaltrial_match.config import ClaudeConfig


class ClaudeClient:
    def __init__(self, config: ClaudeConfig) -> None:
        self._config = config
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise ValueError(f"Missing env var: {config.api_key_env}")
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def tool_use(
        self,
        *,
        model: str,
        system: str,
        user_message: str,
        tools: list[dict[str, Any]],
        tool_name: str,
    ) -> dict[str, Any] | None:
        """Call Claude with a tool, return the tool input dict or None on failure."""
        response = await self._client.messages.create(
            model=model,
            max_tokens=self._config.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_message}],
            tools=tools,  # type: ignore[arg-type]
            tool_choice={"type": "auto"},
            timeout=self._config.timeout_seconds,
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                return block.input  # type: ignore[return-value]
        return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def complete(self, *, model: str, system: str, user_message: str) -> str:
        """Plain text completion."""
        response = await self._client.messages.create(
            model=model,
            max_tokens=self._config.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_message}],
            timeout=self._config.timeout_seconds,
        )
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""

    @property
    def fast_model(self) -> str:
        return self._config.fast_model

    @property
    def reasoning_model(self) -> str:
        return self._config.reasoning_model

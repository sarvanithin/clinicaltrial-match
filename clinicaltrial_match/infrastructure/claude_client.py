"""
Async LLM client with dual-backend support.

Default: Anthropic SDK (direct Claude access).
Optional: Martian Gateway (https://api.withmartian.com/v1) — OpenAI-compatible
  router over 200+ models; enabled when MARTIAN_API_KEY is present.

When Martian is active, all calls route through their endpoint using OpenAI
function-calling format. Anthropic tool_use schema is converted automatically.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from clinicaltrial_match.config import ClaudeConfig

if TYPE_CHECKING:
    from clinicaltrial_match.config import MartianConfig


def _anthropic_tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic tool schemas to OpenAI function-calling format."""
    result = []
    for t in tools:
        result.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
        )
    return result


class ClaudeClient:
    """LLM client that transparently routes to Anthropic or Martian Gateway."""

    def __init__(self, config: ClaudeConfig, martian_config: MartianConfig | None = None) -> None:
        self._config = config
        self._martian_config = martian_config

        self._martian: Any = None
        self._martian_models: dict[str, str] = {}

        # Build Martian client first — if active, Anthropic key is not required
        if martian_config:
            martian_key = os.environ.get(martian_config.api_key_env, "")
            if martian_key:
                import openai  # lazy import — only needed when Martian is active

                self._martian = openai.AsyncOpenAI(
                    base_url=martian_config.base_url,
                    api_key=martian_key,
                )
                self._martian_models = {
                    config.fast_model: martian_config.fast_model,
                    config.reasoning_model: martian_config.reasoning_model,
                }

        # Anthropic client — only required when Martian is not active
        api_key = os.environ.get(config.api_key_env, "")
        if not api_key and self._martian is None:
            raise ValueError(f"Missing env var: {config.api_key_env}. Set ANTHROPIC_API_KEY or MARTIAN_API_KEY.")
        self._client = anthropic.AsyncAnthropic(api_key=api_key or "unused") if api_key else None

    def _martian_model(self, anthropic_model: str) -> str:
        """Map an Anthropic model name to its Martian Gateway equivalent."""
        return self._martian_models.get(anthropic_model, anthropic_model)

    # ------------------------------------------------------------------
    # tool_use — structured extraction via function calling
    # ------------------------------------------------------------------

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
        """Call LLM with a tool; return the tool input dict or None on failure."""
        if self._martian is not None:
            return await self._martian_tool_use(
                model=model,
                system=system,
                user_message=user_message,
                tools=tools,
                tool_name=tool_name,
            )
        return await self._anthropic_tool_use(
            model=model,
            system=system,
            user_message=user_message,
            tools=tools,
            tool_name=tool_name,
        )

    async def _anthropic_tool_use(
        self,
        *,
        model: str,
        system: str,
        user_message: str,
        tools: list[dict[str, Any]],
        tool_name: str,
    ) -> dict[str, Any] | None:
        if self._client is None:
            raise RuntimeError("Anthropic client is not configured — set ANTHROPIC_API_KEY")
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

    async def _martian_tool_use(
        self,
        *,
        model: str,
        system: str,
        user_message: str,
        tools: list[dict[str, Any]],
        tool_name: str,
    ) -> dict[str, Any] | None:
        oai_tools = _anthropic_tools_to_openai(tools)
        response = await self._martian.chat.completions.create(
            model=self._martian_model(model),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            tools=oai_tools,
            tool_choice="auto",
            timeout=self._config.timeout_seconds,
        )
        message = response.choices[0].message
        if message.tool_calls:
            for tc in message.tool_calls:
                if tc.function.name == tool_name:
                    return json.loads(tc.function.arguments)
        return None

    # ------------------------------------------------------------------
    # complete — plain text generation
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def complete(self, *, model: str, system: str, user_message: str) -> str:
        """Plain text completion."""
        if self._martian is not None:
            return await self._martian_complete(model=model, system=system, user_message=user_message)
        return await self._anthropic_complete(model=model, system=system, user_message=user_message)

    async def _anthropic_complete(self, *, model: str, system: str, user_message: str) -> str:
        if self._client is None:
            raise RuntimeError("Anthropic client is not configured — set ANTHROPIC_API_KEY")
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

    async def _martian_complete(self, *, model: str, system: str, user_message: str) -> str:
        response = await self._martian.chat.completions.create(
            model=self._martian_model(model),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            timeout=self._config.timeout_seconds,
        )
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fast_model(self) -> str:
        return self._config.fast_model

    @property
    def reasoning_model(self) -> str:
        return self._config.reasoning_model

    @property
    def backend(self) -> str:
        return "martian" if self._martian is not None else "anthropic"

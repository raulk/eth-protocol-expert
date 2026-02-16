"""Unified LLM completion via LiteLLM (routes to any provider)."""

from dataclasses import dataclass

from litellm import acompletion


@dataclass
class CompletionResponse:
    """Normalized response from any LLM provider."""

    text: str
    input_tokens: int
    output_tokens: int
    model: str


async def call_llm(
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 2048,
) -> CompletionResponse:
    """Call any LLM through LiteLLM's unified interface.

    Model ID conventions (handled by LiteLLM):
      - claude-*          -> Anthropic (ANTHROPIC_API_KEY)
      - gemini/*          -> Google Gemini (GEMINI_API_KEY)
      - openrouter/*      -> OpenRouter (OPENROUTER_API_KEY)
      - gpt-*, o1-*, etc. -> OpenAI (OPENAI_API_KEY)
      - and many more: https://docs.litellm.ai/docs/providers
    """
    response = await acompletion(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )

    choice = response.choices[0]
    usage = response.usage

    return CompletionResponse(
        text=choice.message.content,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        model=model,
    )

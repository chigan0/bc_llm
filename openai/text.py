from langfuse.openai import AsyncAzureOpenAI, AsyncOpenAI

from .dto import LLMConfig
from .base import OpenAIBase


class OpenAITextModel(OpenAIBase):
    def __init__(self, api_key: str, config: LLMConfig) -> None:
        client = AsyncOpenAI(api_key=api_key)
        super().__init__(config=config, client=client)


class AzureTextModel(OpenAIBase):
    def __init__(self, api_key: str, azure_endpoint: str, config: LLMConfig) -> None:
        client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version="2025-04-01-preview",
        )
        super().__init__(config=config, client=client)


__all__ = ("OpenAITextModel", "AzureTextModel")

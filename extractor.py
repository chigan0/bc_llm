from typing import Any, ClassVar, Mapping, Optional, Type

from pydantic import BaseModel

from bc_llm.bc_openai.base import OpenAIBase
from bc_llm.bc_openai.text import AzureTextModel
from bc_llm.bc_openai.models import OpenAiModelEnum
from bc_llm.bc_openai.dto import LLMConfig
from bc_llm.prompt_manager import PromptManager


class BaseLLMExtractor:
    model_class: ClassVar[Type[OpenAIBase]] = AzureTextModel
    config: ClassVar[LLMConfig] = LLMConfig(
        model_name=OpenAiModelEnum.GPT_4_1_MINI,
        timeout=15.0,
        temperature=0.0
    )

    prompt_name: ClassVar[Optional[str]] = None
    instructions: ClassVar[Optional[str]] = None
    response_model: ClassVar[Type[BaseModel]]

    _prompt_cache: ClassVar[Optional[str]] = None
    _model_instance: ClassVar[Optional[OpenAIBase]] = None

    @classmethod
    def _get_model(cls) -> OpenAIBase:
        if cls._model_instance is None:
            cls._model_instance = cls.model_class(cls.config)
        return cls._model_instance

    @classmethod
    def default_prompt_vars(cls) -> Mapping[str, Any]:
        return {}

    @classmethod
    async def _get_prompt_template(cls) -> str:
        if cls._prompt_cache is None:
            if cls.prompt_name:
                cls._prompt_cache = await PromptManager.get_prompt(cls.prompt_name)
            elif cls.instructions:
                cls._prompt_cache = cls.instructions
            else:
                raise ValueError(f"{cls.__name__}: должен быть указан prompt_name или instructions")

        return cls._prompt_cache

    @classmethod
    async def _render_prompt(cls, prompt_vars: Optional[Mapping[str, Any]]) -> str:
        template = await cls._get_prompt_template()

        merged: dict[str, Any] = dict(cls.default_prompt_vars())
        if prompt_vars:
            merged.update(prompt_vars)

        if merged:
            return PromptManager.render_template(template, merged)

        return template

    @classmethod
    async def extract(
        cls,
        user_message: str,
        *,
        prompt_vars: Optional[Mapping[str, Any]] = None,
        parse_kwargs: Optional[Mapping[str, Any]] = None,
        response_model: Optional[Type[BaseModel]] = None,
        instructions: Optional[str] = None,
    ):
        prompt = instructions if instructions is not None else await cls._render_prompt(prompt_vars)
        output_model = response_model or cls.response_model

        model = cls._get_model()
        response = await model.parse(
            new_input=user_message,
            instructions=prompt,
            text_format=output_model,
            **(dict(parse_kwargs) if parse_kwargs else {}),
        )

        return response.content

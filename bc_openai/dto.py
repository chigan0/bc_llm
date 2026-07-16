from dataclasses import dataclass, asdict
from typing import Union

from openai import Omit
from openai._types import omit
from openai.types import ReasoningEffort

from .models import OpenAiModelEnum, verbosity_vals


@dataclass
class LLMConfig:
    """
    Конфигурация LLM клиента.

    effort/verbosity/temperature = omit для параметров, которые не нужно передавать
    """
    model_name: OpenAiModelEnum
    effort: Union[ReasoningEffort, Omit, None] = omit
    verbosity: Union[verbosity_vals, Omit, None] = omit
    temperature: Union[float, Omit, None] = omit
    timeout: float = 60.0
    retry_on_timeout_attempts: int = 2
    retry_on_timeout_delay: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

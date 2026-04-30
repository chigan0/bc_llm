from enum import Enum
from dataclasses import dataclass
from typing import Union, TypedDict, Optional, Literal, Dict, List, TypeVar, Generic

from pydantic import BaseModel
from openai.types.responses import Response, ResponseFunctionToolCall

verbosity_vals = Literal["low", "medium", "high"]
ModelRespG = TypeVar("T", bound=BaseModel)


class OpenAiModelEnum(Enum):
    # GPT-4.1 family
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_4_1_NANO = "gpt-4.1-nano"

    # GPT-5 family
    GPT_5 = "gpt-5"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_NANO = "gpt-5-nano"

    # GPT-5.1 family
    GPT_5_1 = "gpt-5.1"

    # GPT-5.2 family
    GPT_5_2 = "gpt-5.2"


class TokenUsageDict(TypedDict):
    input: int
    output: int
    cache: int


class ExpenseDetailsDict(TypedDict):
    input: float
    output: float
    cache: float
    total: float


class CostReportDict(TypedDict):
    tokens: TokenUsageDict
    expenses: ExpenseDetailsDict


@dataclass
class ModelResponse(Generic[ModelRespG]):
    content: ModelRespG
    response: Response
    cost: Optional[CostReportDict]
    tool_call: Optional[List[ResponseFunctionToolCall]] = None


model_prices: Dict[OpenAiModelEnum, Dict[str, float]] = {
    # GPT-4.1 family
    OpenAiModelEnum.GPT_4_1: {
        "input": 2.0e-6,    # $2.00 / 1M
        "cache": 0.5e-6,    # $0.50 / 1M
        "output": 8.0e-6,   # $8.00 / 1M
    },
    OpenAiModelEnum.GPT_4_1_MINI: {
        "input": 0.4e-6,    # $0.40 / 1M
        "cache": 0.1e-6,    # $0.10 / 1M
        "output": 1.6e-6,   # $1.60 / 1M
    },
    OpenAiModelEnum.GPT_4_1_NANO: {
        "input": 0.1e-6,    # $0.10 / 1M
        "cache": 0.025e-6,  # $0.025 / 1M
        "output": 0.4e-6,   # $0.40 / 1M
    },

    # GPT-5 family
    OpenAiModelEnum.GPT_5: {
        "input": 1.25e-6,   # $1.25 / 1M
        "cache": 0.125e-6,  # $0.125 / 1M
        "output": 10.0e-6,  # $10.00 / 1M
    },
    OpenAiModelEnum.GPT_5_MINI: {
        "input": 0.25e-6,   # $0.25 / 1M
        "cache": 0.025e-6,  # $0.025 / 1M
        "output": 2.0e-6,   # $2.00 / 1M
    },
    OpenAiModelEnum.GPT_5_NANO: {
        "input": 0.05e-6,   # $0.05 / 1M
        "cache": 0.005e-6,  # $0.005 / 1M
        "output": 0.4e-6,   # $0.40 / 1M
    },

    # GPT-5.1 family
    OpenAiModelEnum.GPT_5_1: {
        "input": 1.25e-6,   # $1.25 / 1M
        "cache": 0.125e-6,  # $0.125 / 1M
        "output": 10.0e-6,  # $10.00 / 1M
    },

    # GPT-5.2 family
    OpenAiModelEnum.GPT_5_2: {
        "input": 1.75e-6,   # $1.75 / 1M
        "cache": 0.175e-6,  # $0.175 / 1M
        "output": 14.0e-6,  # $14.00 / 1M
    },
}

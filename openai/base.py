import json
import asyncio
from typing import Union, List, Dict, Optional, Any, Callable, Coroutine, Type

from openai import APITimeoutError
from openai.types.responses import (
    ResponseUsage,
    ResponseFunctionToolCall,
    Response,
)
from langfuse import get_client
from langfuse.openai import AsyncAzureOpenAI, AsyncOpenAI

from .dto import LLMConfig
from .models import (
    model_prices,
    ModelRespG,
    OpenAiModelEnum,
    ModelResponse,
    CostReportDict,
)


class OpenAIBase:
    def __init__(
        self,
        config: LLMConfig,
        client: Union[AsyncAzureOpenAI, AsyncOpenAI],
    ):
        self.config = config
        self.client = client

    @property
    def model_name(self) -> OpenAiModelEnum:
        return self.config.model_name

    @staticmethod
    def format_tool_output(tool_call: ResponseFunctionToolCall, result: Any) -> Dict:
        return {
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": json.dumps(result, ensure_ascii=False),
        }

    @staticmethod
    def calculate_cost(usage: ResponseUsage, model: OpenAiModelEnum) -> CostReportDict:
        p = model_prices[model]

        cached_tokens = usage.input_tokens_details.cached_tokens
        new_input_tokens = usage.input_tokens - cached_tokens
        output_tokens = usage.output_tokens

        input_expenses = new_input_tokens * p["input"]
        output_expenses = output_tokens * p["output"]
        cache_expenses = cached_tokens * (p.get("cache") or 0.0)

        tokens = {
            "input": new_input_tokens,
            "output": output_tokens,
            "cache": cached_tokens,
        }
        expenses = {
            "input": input_expenses,
            "output": output_expenses,
            "cache": cache_expenses,
            "total": input_expenses + output_expenses + cache_expenses,
        }

        return {
            "tokens": tokens,
            "expenses": expenses,
        }

    def _build_base_params(
        self,
        new_input: Union[str, List[Dict]],
        instructions: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "model": self.model_name.value,
            "input": new_input,
            "instructions": instructions,
            "reasoning": {"effort": self.config.effort},
            "temperature": self.config.temperature,
        }

    def _build_request_params(
        self,
        new_input: Union[str, List[Dict]],
        instructions: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        is_json: bool = False,
        json_schema: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        params = self._build_base_params(new_input, instructions)

        text_conf: Dict[str, Any] = {"verbosity": self.config.verbosity}

        if json_schema:
            text_conf["format"] = {
                "type": "json_schema",
                "strict": True,
                "name": "data_extractor",
                "schema": json_schema,
            }
        elif is_json:
            text_conf["format"] = {"type": "json_object"}

        params["text"] = text_conf

        if tools:
            params["tools"] = tools

        return params

    async def _execute_request(
        self,
        api_call: Callable[..., Coroutine[Any, Any, Response]],
        timeout: Optional[float],
        retry_on_timeout_attempts: Optional[int],
        retry_on_timeout_delay: Optional[float],
        **kwargs: Any,
    ) -> Response:
        timeout = timeout if timeout is not None else self.config.timeout

        attempts = (
            retry_on_timeout_attempts
            if retry_on_timeout_attempts is not None
            else self.config.retry_on_timeout_attempts
        )
        delay_base = (
            retry_on_timeout_delay
            if retry_on_timeout_delay is not None
            else self.config.retry_on_timeout_delay
        )

        langfuse = get_client()
        trace_id = langfuse.get_current_trace_id()

        if trace_id:
            with langfuse.start_as_current_span(name=f"LLM: {self.model_name.value}") as span:
                instruction = kwargs.get("instructions")
                if instruction:
                    span.update(metadata={"instruction": instruction})
                return await self._do_request(api_call, timeout, attempts, delay_base, **kwargs)
        else:
            return await self._do_request(api_call, timeout, attempts, delay_base, **kwargs)

    async def _do_request(
        self,
        api_call: Callable[..., Coroutine[Any, Any, Response]],
        timeout: float,
        attempts: int,
        delay_base: float,
        **kwargs: Any,
    ) -> Response:
        last_exception: Optional[Exception] = None

        for attempt in range(attempts + 1):
            try:
                resp: Response = await api_call(timeout=timeout, **kwargs)
                return resp

            except APITimeoutError as e:
                last_exception = e
                if attempt < attempts:
                    delay = delay_base * (2**attempt)
                    await asyncio.sleep(delay)

        raise last_exception  # type: ignore

    async def create(
        self,
        new_input: Union[str, List[Dict]],
        *,
        instructions: Optional[str] = None,
        is_json: bool = False,
        json_schema: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        timeout: Optional[float] = None,
        retry_on_timeout_attempts: Optional[int] = None,
        retry_on_timeout_delay: Optional[float] = None,
    ) -> ModelResponse:
        request_params = self._build_request_params(
            new_input=new_input,
            instructions=instructions,
            tools=tools,
            is_json=is_json,
            json_schema=json_schema,
        )

        response = await self._execute_request(
            api_call=self.client.responses.create,
            timeout=timeout,
            retry_on_timeout_attempts=retry_on_timeout_attempts,
            retry_on_timeout_delay=retry_on_timeout_delay,
            **request_params,
        )

        cost_dict: Optional[CostReportDict] = None
        if response.usage:
            cost_dict = self.calculate_cost(response.usage, self.model_name)

        if is_json or json_schema:
            content: Union[str, dict] = json.loads(response.output_text)
        else:
            content = response.output_text

        tool_calls = [
            item for item in response.output if isinstance(item, ResponseFunctionToolCall)
        ] or None

        return ModelResponse(
            response=response,
            content=content,
            cost=cost_dict,
            tool_call=tool_calls,
        )

    async def parse(
        self,
        new_input: Union[str, List[Dict]],
        *,
        instructions: Optional[str] = None,
        text_format: Type[ModelRespG],
        timeout: Optional[float] = None,
        retry_on_timeout_attempts: Optional[int] = None,
        retry_on_timeout_delay: Optional[float] = None,
        **kwargs
    ) -> ModelResponse[ModelRespG]:
        params = self._build_base_params(new_input, instructions)
        params["text_format"] = text_format

        response = await self._execute_request(
            api_call=self.client.responses.parse,
            timeout=timeout,
            retry_on_timeout_attempts=retry_on_timeout_attempts,
            retry_on_timeout_delay=retry_on_timeout_delay,
            **params,
            **kwargs
        )

        cost_dict: Optional[CostReportDict] = None

        if response.usage:
            cost_dict = self.calculate_cost(response.usage, self.model_name)

        tool_calls = [
            item for item in response.output if isinstance(item, ResponseFunctionToolCall)
        ] or None

        return ModelResponse[ModelRespG](
            response=response,
            content=response.output_parsed,
            cost=cost_dict,
            tool_call=tool_calls,
        )

    async def __call__(self, *args, **kwargs) -> ModelResponse:
        return await self.create(*args, **kwargs)

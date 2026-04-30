from functools import wraps
from typing import Callable, Any

from langfuse import get_client

from bc_llm.settings import get_llm_settings


def client_trace(trace_name: str, use_inst: bool = False):
    def decorator(func: Callable[..., Any]):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            langfuse = get_client()

            with langfuse.start_as_current_span(name=trace_name) as trace:
                __kwargs = {
                    "tags": [get_llm_settings().product],
                }

                if use_inst:
                    instance = args[0]  # self
                    __kwargs = {
                        **__kwargs,
                        "user_id": instance.client_id,
                        "session_id": instance.session_id,
                    }

                trace.update_trace(**__kwargs)
                return await func(*args, trace=trace, **kwargs)

        return async_wrapper

    return decorator

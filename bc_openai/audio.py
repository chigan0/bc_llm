from typing import Dict, Tuple

from openai import OpenAI
from openai.types.audio import TranscriptionVerbose


class OpenAIAudioModel:
    """
    TODO: не реализовано. Черновик под транскрипцию аудио (whisper-1 / gpt-4o-transcribe).

    Известные проблемы прежней версии, которые нужно поправить при реализации:
    - __call__ использует синхронный OpenAI-клиент, тогда как весь остальной bc_llm асинхронный
      (см. OpenAITextModel/AzureTextModel) — нужен AsyncOpenAI.
    - calculate_expenses никогда не вызывался, стоимость/длительность были захардкожены нулями.
    - calculate_expenses был неверно типизирован (Tuple[Dict, Dict] вместо Tuple[float, float]).
    """

    model_prices = {"whisper-1": {"input": 0.006}, "gpt-4o-transcribe": {"input": 0.006}}

    def __init__(self, api_key: str, model_name: str) -> None:
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def calculate_expenses(self, response: TranscriptionVerbose) -> Tuple[float, float]:
        # TODO: реализовать и вызывать из __call__
        ...

    def __call__(self, file_location: str) -> Dict:
        # TODO: реализовать через async-клиент и подключить calculate_expenses
        ...

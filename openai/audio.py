from typing import Dict, Tuple

from openai import OpenAI
from openai.types.audio import TranscriptionVerbose


class OpenAIAudioModel:
    model_prices = {"whisper-1": {"input": 0.006}, "gpt-4o-transcribe": {"input": 0.006}}

    def __init__(self, api_key: str, model_name: str) -> None:
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def calculate_expenses(self, response: TranscriptionVerbose) -> Tuple[Dict, Dict]:
        return response.duration, response.duration / 60 * self.model_prices[self.model_name]["input"]

    def __call__(self, file_location: str) -> Dict:
        with open(file_location, "rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                model=self.model_name,
                file=audio_file,
            )
        return {"content": response.text, "expenses": {"duration": 0, "expenses": 0}}

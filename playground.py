"""
Ручной прогон bc_llm. Запуск:

    cd /home/haron_66/projects/ram
    bc_llm/.venv/bin/python -m bc_llm.playground

(или просто `python playground.py` из bc_llm/ — sys.path поправлен ниже).
Подставь свой ключ в API_KEY перед запуском.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel

from bc_llm.bc_openai import LLMConfig, OpenAiModelEnum, OpenAITextModel

API_KEY = "sk-"  # <-- подставь свой ключ


class Person(BaseModel):
    name: str
    age: int


async def main() -> None:
    config = LLMConfig(
        model_name=OpenAiModelEnum.GPT_4_1_NANO,
        temperature=0.0,
        timeout=30.0,
    )
    model = OpenAITextModel(api_key=API_KEY, config=config)

    print("=== create() — свободный текст ===")
    response = await model.create("Привет! Кто ты? Ответь одним предложением.")
    print("content:", response.content)
    print("cost:", response.cost)

    print("\n=== create() — JSON schema ===")
    response = await model.create(
        "Извлеки имя и возраст: Иван, 32 года",
        json_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
            "additionalProperties": False,
        },
    )
    print("content:", response.content)
    print("cost:", response.cost)

    print("\n=== parse() — pydantic ===")
    response = await model.parse(
        "Извлеки имя и возраст: Иван, 32 года",
        instructions="Ты извлекаешь структурированные данные из текста.",
        text_format=Person,
    )
    print("content:", response.content)
    print("cost:", response.cost)


if __name__ == "__main__":
    asyncio.run(main())

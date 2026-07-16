# bc_llm

Единая обёртка над OpenAI/Azure OpenAI (Responses API) для всех LLM-агентов проекта:
конфиг модели, ретраи, подсчёт стоимости запроса и менеджер промптов с поддержкой
продукт-специфичных шаблонов.

Цель — чтобы все агенты (в т.ч. написанные разными людьми) были устроены одинаково:
одна точка конфигурации модели, один способ считать деньги.

## Структура

```
bc_llm/
  bc_openai/
    dto.py       — LLMConfig (конфиг модели)
    models.py    — OpenAiModelEnum, цены на токены, ModelResponse
    base.py      — OpenAIBase: create() / parse(), ретраи, подсчёт стоимости
    text.py      — OpenAITextModel, AzureTextModel (наследники OpenAIBase)
    audio.py     — заготовка под транскрипцию, НЕ РЕАЛИЗОВАНО (см. TODO в файле)
  prompt_manager/
    manager.py   — PromptManager: чтение/кэш/рендер промптов из products/<Product>/prompts
  extractor.py   — BaseLLMExtractor: базовый класс для агентов-экстракторов
  settings.py    — LLMSettings (.env), сейчас только `product`
```

## Быстрый старт

```python
from bc_llm.bc_openai import LLMConfig, OpenAiModelEnum, OpenAITextModel

config = LLMConfig(
    model_name=OpenAiModelEnum.GPT_5_MINI,
    temperature=0.0,
    timeout=30.0,
    retry_on_timeout_attempts=2,   # ретраи только на APITimeoutError
    retry_on_timeout_delay=1.0,    # экспоненциальный backoff: delay * 2**attempt
)

model = OpenAITextModel(api_key="sk-...", config=config)

response = await model.create("Привет! Кто ты?")
print(response.content)   # str
print(response.cost)      # {"tokens": {...}, "expenses": {...}} или None, если usage не пришёл
```

Azure — тот же интерфейс, другой конструктор:

```python
from bc_llm.bc_openai import AzureTextModel

model = AzureTextModel(
    api_key="...",
    azure_endpoint="https://<resource>.openai.azure.com",
    config=config,
)
```

`model_name` — только значения из `OpenAiModelEnum` (сейчас: GPT-4.1 / 4.1-mini / 4.1-nano,
GPT-5 / 5-mini / 5-nano, GPT-5.1, GPT-5.2). У каждой модели своя цена за токен в
`bc_openai/models.py:model_prices` — при добавлении новой модели цену нужно прописать там же,
иначе `calculate_cost` упадёт с `KeyError`.

## Структурированный вывод

Два способа получить не просто текст, а структуру.

**1. JSON Schema напрямую через `create()`:**

```python
response = await model.create(
    "Извлеки имя и возраст: Иван, 32 года",
    json_schema={
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
        "additionalProperties": False,
    },
)
response.content  # dict: {"name": "Иван", "age": 32}
```

**2. Pydantic-модель через `parse()`** (предпочтительный способ для агентов):

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

response = await model.parse(
    "Извлеки имя и возраст: Иван, 32 года",
    text_format=Person,
)
response.content  # Person(name="Иван", age=32) — уже провалидированный объект
```

## Паттерн агента: `BaseLLMExtractor`

Стандартный способ оформить LLM-агента, который на вход берёт текст и отдаёт pydantic-объект.
Промпт хранится отдельным файлом (не в коде), модель и её конфиг задаются один раз на класс.

```python
# agents/person_extractor.py
from pydantic import BaseModel
from bc_llm.extractor import BaseLLMExtractor
from bc_llm.bc_openai import LLMConfig, OpenAiModelEnum

class Person(BaseModel):
    name: str
    age: int

class PersonExtractor(BaseLLMExtractor):
    config = LLMConfig(model_name=OpenAiModelEnum.GPT_4_1_MINI, temperature=0.0, timeout=15.0)
    prompt_name = "person_extractor.txt"   # products/<Product>/prompts/person_extractor.txt
    response_model = Person

    @classmethod
    def default_prompt_vars(cls) -> dict:
        return {"language": "ru"}  # подставляются в {{language}} внутри промпта


person = await PersonExtractor.extract(
    "Иван, 32 года",
    prompt_vars={"language": "ru"},  # переопределяет default_prompt_vars по ключам
)
```

Что важно:

- `prompt_name` **или** `instructions` — обязательно одно из двух (иначе `ValueError`).
  `instructions` — просто строка-промпт прямо в коде, `prompt_name` — файл через `PromptManager`.
- Промпт и инстанс модели кэшируются на уровне класса (`_prompt_cache`, `_model_instance`),
  повторный `.extract()` не создаёт новый клиент и не перечитывает файл.
- Каждый новый агент — это **новый класс**, наследник `BaseLLMExtractor`, а не параметры функции.
  Так все агенты выглядят одинаково и их легко найти/сравнить.

## Промпты: `PromptManager`

Промпты — обычные текстовые файлы с плейсхолдерами `{{var}}` или `{{obj.field}}`:

```
products/
  Common/
    prompts/
      person_extractor.txt
  MyProduct/
    prompts/
      person_extractor.txt   # переопределяет Common для этого продукта
```

`PromptManager.get_prompt(name)` сначала ищет файл в `products/<product>/prompts/<name>`
(значение `product` берётся из `LLMSettings`, т.е. из `.env` / переменной окружения `PRODUCT`),
и если не находит — падает обратно на `products/Common/prompts/<name>`.

Кэш прогревается по `mtime` файла — поправили промпт на диске, следующий вызов подхватит
новую версию без рестарта. Ручной сброс: `PromptManager.clear_cache()`.

```python
from bc_llm.prompt_manager import PromptManager

text = await PromptManager.get_and_render("person_extractor.txt", {"language": "ru"})
```

## Стоимость запроса

`response.cost` (у `create()`/`parse()`) — это:

```python
{
    "tokens": {"input": int, "output": int, "cache": int},
    "expenses": {"input": float, "output": float, "cache": float, "total": float},  # USD
}
```

Кэшированные входные токены считаются отдельно и по своей (более низкой) цене.
Если `response.usage` не пришёл от API — `cost` будет `None`, это нужно проверять перед чтением.

## Чек-лист для нового агента

1. Определить, нужен ли structured output (`parse()` + pydantic) или просто текст (`create()`).
2. Если это "текст → структура" — сделать класс-наследник `BaseLLMExtractor`, а не писать
   вызов модели руками.
3. Промпт — в файл `products/<Product или Common>/prompts/...`, не строкой в коде
   (кроме простых случаев через `instructions`).
4. Модель/температуру/effort задавать через `LLMConfig` на классе, не как magic-константы внутри метода.

## Известные ограничения

- `bc_openai/audio.py` — не реализован, это заготовка. См. TODO в файле перед использованием.
- Поддерживается только OpenAI **Responses API** (`responses.create` / `responses.parse`),
  не Chat Completions.

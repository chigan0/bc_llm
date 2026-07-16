# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`bc_llm` is a shared internal library, not a standalone app. It is meant to be vendored/installed
as a subpackage inside a parent project and imported as `bc_llm.*` (all internal imports already
use the absolute `bc_llm.xxx` form — see `extractor.py`). It wraps the OpenAI
**Responses API** (both plain OpenAI and Azure OpenAI) with:

- a single `LLMConfig`/model-enum surface so every agent configures models the same way,
- automatic retry-with-backoff on request timeouts,
- per-request token/dollar cost accounting,
- a file-based prompt template system scoped by "product".

Full usage examples (quick start, structured output, the extractor pattern, prompts, cost shape)
live in [README.md](README.md) — read that before writing a new agent, this file is about the
internals, not usage.

## Development commands

There is currently no build tooling, linter config, or test suite in this repo (no `pyproject.toml`,
`setup.py`, `Makefile`, or `tests/`). Dependencies are listed with minimum versions in [req.txt](req.txt) and installed
with `pip install -r req.txt` into the parent project's environment. When adding new functionality,
check whether the parent project has since added test/lint tooling before assuming there is none.

## Architecture

The package is layered; higher layers depend on lower ones and agents should be built at the
top layer, not by reaching into `bc_openai/` directly.

1. **`bc_openai/` — LLM client layer.**
   - `dto.py`: `LLMConfig`, the one place model/temperature/effort/verbosity/timeout/retry
     settings live. Unset optional fields default to the `openai` SDK's `omit` sentinel so they're
     not sent in the request at all (not `None` — `None` and "omit" are different on the wire).
   - `models.py`: `OpenAiModelEnum` (only models listed here are valid `model_name` values) and
     `model_prices`, a hardcoded $/token table keyed by that enum. Adding a model to the enum
     without adding a price entry makes `OpenAIBase.calculate_cost` raise `KeyError` at runtime,
     not at import time — always add both together.
   - `base.py`: `OpenAIBase` — the shared request/response machinery for both backends:
     `_build_request_params` (assembles the Responses API payload, handles `json_schema`/`is_json`),
     `_do_request`/`_execute_request` (retry loop, only catches `APITimeoutError`), `create()`
     (free-form/JSON output) and `parse()` (typed output via a pydantic `text_format`). Both
     return a `ModelResponse` with `.content`, `.cost`, `.tool_call`.
   - `text.py`: `OpenAITextModel` / `AzureTextModel` — thin constructors that build the right
     `AsyncOpenAI`/`AsyncAzureOpenAI` client and hand it to `OpenAIBase`. This is the intended
     entry point; new backends should follow the same "build a client, pass to `OpenAIBase`" shape.
   - `audio.py`: **not implemented**, a stub with `...` bodies and TODOs. Do not wire agents to it
     as-is — see the TODO comments in the file for what's missing (needs an async client, and
     `calculate_expenses` needs to actually be called).

2. **`prompt_manager/` — prompt templates.**
   `PromptManager` resolves prompt files from `products/<product>/prompts/<name>`, falling back to
   `products/Common/prompts/<name>` when the product-specific file doesn't exist. `<product>` comes
   from `LLMSettings.product` (`settings.py`, reads `PRODUCT` from `.env`). Templates use
   `{{var}}` / `{{obj.field}}` placeholders resolved via `render_template`. Prompts are cached
   in-process keyed by file `mtime`, so editing a prompt file on disk is picked up on next call
   without a restart — `clear_cache()` exists for tests.

3. **`extractor.py` — the agent base class.**
   `BaseLLMExtractor` is the standard shape for a "text in → validated pydantic object out" agent:
   a subclass declares `config` (an `LLMConfig`), either `prompt_name` (loaded via `PromptManager`)
   or `instructions` (inline string), and `response_model`. `_get_model()`/`_get_prompt_template()`
   lazily build and cache the model instance and rendered prompt **per subclass** (they're
   `ClassVar`s but assignment happens on `cls`, so sibling subclasses don't clobber each other's
   cache). New agents should be new `BaseLLMExtractor` subclasses, not ad-hoc calls into `bc_openai/`.

4. **`settings.py`** — just `LLMSettings.product`, read from `.env`, driving the prompt-manager
   product routing above. `get_llm_settings()` is `lru_cache`d.

## Gotchas specific to this repo

- The LLM client subpackage is named `bc_openai/`, not `openai/` — it was renamed specifically to
  avoid shadowing the third-party `openai` PyPI package that `base.py`/`text.py`/`models.py` import
  from directly (`from openai import APITimeoutError, AsyncOpenAI, ...`). Keep it that way; do not
  rename it back to `openai/`.
- Only the Responses API is supported (`client.responses.create` / `.parse`), not Chat Completions.
  Don't add Chat Completions calls into `OpenAIBase` — build a parallel path if that's ever needed.
- Versioning lives in a plain [VERSION](VERSION) file (currently `0.1.0`), not in `__init__.py`.

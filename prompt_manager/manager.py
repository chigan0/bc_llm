import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import aiofiles

from bc_llm.settings import get_llm_settings
from .dto import CacheEntry


async def _open_file_async(path: str, is_json: bool = False):
    async with aiofiles.open(path, "r", encoding="utf-8") as file:
        content = await file.read()
    return json.loads(content) if is_json else content


class PromptManager:
    COMMON_ROOT = Path("products/Common")

    # {{var}}  или {{ user.name }}
    _braces_pattern = re.compile(r"\{\{\s*([A-Za-z0-9_.]+)\s*\}\}")

    _cache: Dict[str, CacheEntry] = {}

    @classmethod
    def _product_root(cls) -> Path:
        return Path(f"products/{get_llm_settings().product}")

    @classmethod
    def _resolve_path(cls, name: str) -> Path:
        product_path = cls._product_root() / f"prompts/{name}"
        if product_path.exists():
            return product_path
        return cls.COMMON_ROOT / f"prompts/{name}"

    @classmethod
    async def get_prompt(cls, name: str) -> str:
        path = cls._resolve_path(name)
        current_mtime = path.stat().st_mtime

        cached = cls._cache.get(name)
        if cached and cached.mtime == current_mtime:
            return cached.content

        content = await _open_file_async(path)
        cls._cache[name] = CacheEntry(mtime=current_mtime, content=content)
        return content

    @classmethod
    async def get_json_config(cls, relative_path: str) -> Dict[str, Any]:
        path = cls._product_root() / relative_path
        return await _open_file_async(path, is_json=True)

    @classmethod
    def _resolve_key(cls, data: Mapping[str, Any], key: str) -> Optional[Any]:
        current: Any = data
        for part in key.split("."):
            if isinstance(current, Mapping) and part in current:
                current = current[part]
            else:
                return None
        return current

    @classmethod
    def render_template(
        cls,
        template: str,
        variables: Mapping[str, Any],
    ) -> str:
        def replacer(match: re.Match) -> str:
            key = match.group(1)
            value = cls._resolve_key(variables, key)
            if value is None:
                return match.group(0)
            if isinstance(value, list):
                return ", ".join(map(str, value))
            return str(value)

        return cls._braces_pattern.sub(replacer, template)

    @classmethod
    async def get_and_render(
        cls,
        name: str,
        variables: Mapping[str, Any],
    ) -> str:
        template = await cls.get_prompt(name)
        return cls.render_template(template, variables)

    @classmethod
    def find_placeholders(cls, template: str) -> List[str]:
        return cls._braces_pattern.findall(template)

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()

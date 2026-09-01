"""Clientes de API e funções de consulta ("providers") para cada família de
modelo. Os clientes são inicializados de forma preguiçosa (lazy) e única,
para que o módulo possa ser importado sem exigir todas as chaves de API
presentes (ex.: modo fine-tuned não precisa de GEMINI_API_KEY/DEEPSEEK_API_KEY).
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from functools import lru_cache
from typing import Callable, Coroutine

from google import genai
from openai import AsyncOpenAI, OpenAI

from .config import MODEL_CONTEXT

logger = logging.getLogger(__name__)

ProviderFn = Callable[[str, str], Coroutine[None, None, str]]


@lru_cache(maxsize=1)
def get_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@lru_cache(maxsize=1)
def get_deepseek_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
    )


@lru_cache(maxsize=1)
def get_gemini_client() -> genai.Client:
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


async def ask_openai(model: str, question: str) -> str:
    try:
        resp = await get_openai_client().responses.create(
            model=model, input=question, instructions=MODEL_CONTEXT
        )
        return resp.output_text.strip()
    except Exception as e:
        logger.error(f"OpenAI error ({model}): {e}")
        return " "


async def ask_deepseek(model: str, question: str) -> str:
    try:
        resp = await asyncio.to_thread(
            lambda: get_deepseek_client().chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": MODEL_CONTEXT},
                    {"role": "user", "content": question},
                ],
            )
        )
        content = resp.choices[0].message.content
        return content.strip() if content else " "
    except Exception as e:
        logger.error(f"DeepSeek error ({model}): {e}")
        return " "


async def ask_gemini(model: str, question: str) -> str:
    try:
        resp = await get_gemini_client().aio.models.generate_content(
            model=model,
            contents=question,
            config=genai.types.GenerateContentConfig(
                system_instruction=MODEL_CONTEXT,
            ),
        )
        return resp.text.strip() if resp.text else " "
    except Exception as e:
        logger.error(f"Gemini error ({model}): {e}")
        return " "


PROVIDERS: dict[str, ProviderFn] = {
    "openai": ask_openai,
    "deepseek": ask_deepseek,
    "gemini": ask_gemini,
}


async def retry(fn: ProviderFn, *args, retries: int = 3, delay: float = 2, **kwargs):
    """Reexecuta `fn` em caso de erro, com backoff linear + jitter."""
    for attempt in range(1, retries + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            if attempt == retries:
                logger.exception(f"Error after {retries} retries: {e}")
                return f"Error after {retries} retries: {e}"
            sleep_time = delay * attempt + random.random()
            logger.warning(f"Error: {e}. Retrying in {sleep_time:.1f}s...")
            await asyncio.sleep(sleep_time)


async def aclose_all() -> None:
    """Fecha todos os clientes que já foram instanciados."""
    closers = []
    if get_openai_client.cache_info().currsize:
        closers.append(get_openai_client().close())
    if get_gemini_client.cache_info().currsize:
        closers.append(get_gemini_client().aio.aclose())
    # OpenAI-compatible sync client (DeepSeek) não precisa de close assíncrono.
    if closers:
        await asyncio.gather(*closers, return_exceptions=True)

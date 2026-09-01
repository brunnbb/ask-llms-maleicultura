"""Configuração central: variáveis de ambiente, prompt do sistema e registro
de modelos para cada modo de execução (normal x fine-tuning).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Coroutine

from dotenv import load_dotenv

load_dotenv()

MODEL_CONTEXT = """
Você é um consultor agrícola especializado em produção e manejo de maçãs, com foco em ajudar produtores rurais, considerando o contexto produtivo da região sul do Brasil.
Seu papel é orientar produtores sobre plantio, irrigação, poda, controle de pragas, colheita, comercialização e qualquer outro aspecto da produção de maçãs.
Responda e forneça recomendações baseadas em práticas agrícolas comprovadas e adaptadas às condições dadas.
Responda de forma clara, concisa, curta e prática, em único parágrafo com poucas frases de maneira simples e resumido, sem markdown ou outras formatações.
"""


class Mode(str, Enum):
    NORMAL = "normal"
    FINE_TUNED = "fine-tuned"


@dataclass(frozen=True)
class ModelSpec:
    """Descreve um único modelo a ser consultado."""

    column: str  # nome da coluna no CSV/DataFrame
    model_id: str  # nome/id do modelo a passar para a API
    provider: str  # chave em PROVIDERS (ex.: "openai", "gemini", "deepseek")


# Cada provider expõe uma função async (model_id, question) -> str.
# A implementação real fica em providers.py; aqui só referenciamos a chave
# para manter este módulo livre de dependências pesadas de SDK.
ProviderFn = Callable[[str, str], Coroutine[None, None, str]]


@dataclass(frozen=True)
class RunConfig:
    """Configuração completa de uma execução do pipeline."""

    mode: Mode
    csv_path: str
    models: tuple[ModelSpec, ...]
    concurrency: int  # nº de perguntas processadas em paralelo


def _normal_models() -> tuple[ModelSpec, ...]:
    return (
        ModelSpec("gpt-5", "gpt-5", "openai"),
        ModelSpec("gpt-5-mini", "gpt-5-mini", "openai"),
        ModelSpec("gpt-5-nano", "gpt-5-nano", "openai"),
        ModelSpec("gpt-4.1", "gpt-4.1", "openai"),
        ModelSpec("gpt-4.1-mini", "gpt-4.1-mini", "openai"),
        ModelSpec("gemini-2.5-flash", "gemini-2.5-flash", "gemini"),
        ModelSpec("gemini-2.5-pro", "gemini-2.5-pro", "gemini"),
        ModelSpec("deepseek-chat", "deepseek-chat", "deepseek"),
        ModelSpec("deepseek-reasoner", "deepseek-reasoner", "deepseek"),
    )


def _fine_tuned_models() -> tuple[ModelSpec, ...]:
    # Fine-tuning só existe para a família OpenAI (DeepSeek não tem API de
    # fine-tuning nativa e Gemini não foi incluído no experimento).
    gpt4_1 = os.getenv("FT_MODEL_NAME_GPT_4_1")
    gpt4_1_mini = os.getenv("FT_MODEL_NAME_GPT_4_1_MINI")

    missing = [
        name
        for name, value in (
            ("FT_MODEL_NAME_GPT_4_1", gpt4_1),
            ("FT_MODEL_NAME_GPT_4_1_MINI", gpt4_1_mini),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"Variáveis de ambiente ausentes para o modo fine-tuned: {', '.join(missing)}"
        )

    return (
        ModelSpec("gpt-4.1-ft", gpt4_1, "openai"),
        ModelSpec("gpt-4.1-mini-ft", gpt4_1_mini, "openai"),
    )


def build_run_config(
    mode: Mode,
    csv_path: str | None = None,
    concurrency: int | None = None,
) -> RunConfig:
    """Monta a configuração de execução a partir do modo escolhido,
    aplicando os defaults originais de cada script quando não sobrescritos.
    """
    if mode is Mode.NORMAL:
        return RunConfig(
            mode=mode,
            csv_path=csv_path or "src/perguntas_e_respostas.csv",
            models=_normal_models(),
            concurrency=concurrency or 1,
        )

    if mode is Mode.FINE_TUNED:
        return RunConfig(
            mode=mode,
            csv_path=csv_path or "src/normal_fined_tune.csv",
            models=_fine_tuned_models(),
            concurrency=concurrency or 4,
        )

    raise ValueError(f"Modo desconhecido: {mode}")

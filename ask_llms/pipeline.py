"""Pipeline genérico de processamento do CSV de perguntas, parametrizado
por um RunConfig (modo normal ou fine-tuned).
"""

from __future__ import annotations

import asyncio
import logging

import pandas as pd

from .config import RunConfig
from .providers import PROVIDERS, retry

logger = logging.getLogger(__name__)


async def process_csv(config: RunConfig) -> None:
    df = pd.read_csv(config.csv_path)
    logger.info(f"Processando {len(df)} perguntas do CSV ({config.mode.value})")

    # Garante que todas as colunas de modelo existam e sejam strings
    for spec in config.models:
        if spec.column not in df.columns:
            df[spec.column] = ""
        df[spec.column] = df[spec.column].fillna("").astype(str)

    sem = asyncio.Semaphore(config.concurrency)

    async def query_model(i: int, question: str, spec) -> str:
        # Pula se já houver resposta salva
        existing = str(df.at[i, spec.column]).strip()
        if existing:
            return existing
        fn = PROVIDERS[spec.provider]
        return await retry(fn, spec.model_id, question)

    async def process_row(i: int, question: str) -> None:
        async with sem:
            logger.info(f"→ Pergunta {i + 1}: processando...")

            results = await asyncio.gather(
                *[query_model(i, question, spec) for spec in config.models]
            )

            for spec, result in zip(config.models, results):
                df.at[i, spec.column] = result

            df.to_csv(config.csv_path, index=False, encoding="utf-8")
            logger.info(f"✓ Pergunta {i + 1}: concluída e salva.")

    await asyncio.gather(
        *[
            process_row(i, row["pergunta"].strip())
            for i, row in df.iterrows()
            if pd.notna(row["pergunta"]) and str(row["pergunta"]).strip()
        ]
    )

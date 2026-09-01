"""Ponto de entrada único.

Substitui main.py + fined_tuned.py: o modo de execução é escolhido via
--mode (normal | fine-tuned), mantendo o comportamento e os defaults
originais de cada script (arquivo CSV, concorrência, conjunto de modelos).

Exemplos:
    python main.py --help                             # Menu de ajuda com as flags possíveis
    python main.py                                    # Modo normal (default)
    python main.py --mode fine-tuned                  # Modo fine-tuning (só OpenAI)
    python main.py --mode normal --csv data/test.csv  # Lê outro csv dentro da pasta data
    python main.py --mode fine-tuned --concurrency 4  # Número de threads concorrentes
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

from ask_llms.config import Mode, build_run_config
from ask_llms.pipeline import process_csv
from ask_llms.providers import aclose_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consulta LLMs para o CSV de perguntas de maleicultura."
    )
    _ = parser.add_argument(
        "--mode",
        choices=[m.value for m in Mode],
        default=Mode.NORMAL.value,
        help="normal: todos os modelos base (OpenAI, Gemini, DeepSeek). "
        "fine-tuned: apenas os modelos OpenAI fine-tuned.",
    )
    _ = parser.add_argument(
        "--csv",
        default=None,
        help="Caminho do CSV a processar (default depende do modo).",
    )
    _ = parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Nº de perguntas processadas em paralelo (default depende do modo).",
    )
    return parser.parse_args()


async def run(mode: Mode, csv_path: str | None, concurrency: int | None) -> None:
    start = time.perf_counter()
    config = build_run_config(mode, csv_path=csv_path, concurrency=concurrency)
    try:
        await process_csv(config)
    except Exception as e:
        logger.exception(f"Erro no processamento: {e}")
    finally:
        await aclose_all()
    elapsed = time.perf_counter() - start
    logger.info(f"Tempo total: {elapsed:.2f}s")


def main() -> None:
    args = parse_args()
    asyncio.run(run(Mode(args.mode), args.csv, args.concurrency))


if __name__ == "__main__":
    main()

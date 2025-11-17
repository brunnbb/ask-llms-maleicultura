import os
import asyncio
import time
import random
import logging
from typing import Callable

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from google import genai
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()

MODEL_CONTEXT = (
"""
Você é um consultor agrícola especializado em produção e manejo de maçãs, com foco em ajudar produtores rurais, considerando o contexto produtivo da região sul do Brasil.
Seu papel é orientar produtores sobre plantio, irrigação, poda, controle de pragas, colheita, comercialização e qualquer outro aspecto da produção de maçãs.
Responda e forneça recomendações baseadas em práticas agrícolas comprovadas e adaptadas às condições dadas.
Responda de forma clara, concisa, curta e prática, em único parágrafo com poucas frases de maneira simples e resumido, sem markdown ou outras formatações.
"""
)

# Initialize clients globally
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
deepseek_client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# LLM wrappers
async def ask_openai(model: str, question: str) -> str:
    try:
        resp = await openai_client.responses.create(
            model=model,
            input=question,
            instructions=MODEL_CONTEXT
        )
        return resp.output_text.strip()
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        return " " 

async def ask_deepseek(model: str, question: str) -> str:   
    try:
        resp = await asyncio.to_thread(
            lambda: deepseek_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": MODEL_CONTEXT},
                    {"role": "user", "content": question},
                ],
            )
        )
        return resp.choices[0].message.content.strip() if resp.choices[0].message.content else " "
    except Exception as e:
        logger.error(f"DeepSeek error: {e}")
        return " " 

async def ask_gemini(model: str, question: str) -> str:
    try:
        resp = await gemini_client.aio.models.generate_content(
            model=model,
            contents=question,
            config=genai.types.GenerateContentConfig(
                system_instruction=MODEL_CONTEXT,
                ),
        )
        return resp.text.strip() if resp.text else " "
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return " " 

# Retry logic in case of network errors or rate limits
async def retry(fn: Callable, *args, retries=3, delay=2, **kwargs):
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

# Model mapping
MODELS = {
    "gpt-5": ask_openai,
    "gpt-5-mini": ask_openai,
    "gpt-5-nano": ask_openai,
    "gemini-2.5-flash": ask_gemini,
    "gemini-2.5-pro": ask_gemini,
    "deepseek-chat": ask_deepseek,
    "deepseek-reasoner": ask_deepseek,
}

# CSV pipeline
async def process_csv(filename: str):
    df = pd.read_csv(filename)
    logger.info(f"Processando {len(df)} perguntas do CSV")
    
    # Ensure all model columns exist and are strings    
    for name in MODELS.keys():
        if name not in df.columns:
            df[name] = ""
        df[name] = df[name].fillna("").astype(str)

    # Semaphore to limit concurrency
    sem = asyncio.Semaphore(1)

    # Process each row
    async def process_row(i, question):
        async with sem:
            logger.info(f"→ Pergunta {i+1}: processando...")

            # Query each model
            async def query_model(model, fn):
                # Check if already answered, if so, skip
                if str(df.at[i, model]).strip():
                    return df.at[i, model]
                # Ask with retry
                return await retry(fn, model, question)

            # List comprehension to gather all model responses
            results = await asyncio.gather(*[query_model(model, fn) for model, fn in MODELS.items()])

            # Update DataFrame with results
            for model, result in zip(MODELS.keys(), results):
                df.at[i, model] = result

            # Save each row results
            df.to_csv(filename, index=False, encoding="utf-8")
            logger.info(f"✓ Pergunta {i+1}: concluída e salva.")
    
    # Process all rows concurrently
    await asyncio.gather(*[
        process_row(i, row["pergunta"].strip())
        for i, row in df.iterrows()
        if pd.notna(row["pergunta"]) and str(row["pergunta"]).strip()
    ])

# Entry point
async def main():
    start = time.perf_counter()
    try:
        await process_csv("src/perguntas_e_respostas.csv")
    except Exception as e:
        logger.exception(f"Erro no processamento: {e}")
    finally:
        await asyncio.gather(
            openai_client.close(),
            gemini_client.aio.aclose(),
            return_exceptions=True
        )
    elapsed = time.perf_counter() - start
    logger.info(f"Tempo total: {elapsed:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())

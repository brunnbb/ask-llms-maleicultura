# ask-llms-maleicultura

Pipeline que envia um conjunto de perguntas sobre **maleicultura** (cultivo de macieiras) para múltiplos LLMs e salva as respostas em um CSV, permitindo comparar o desempenho de diferentes modelos.
Suporta dois modos de execução:

- **normal** — consulta todos os modelos base (OpenAI, Google Gemini, DeepSeek).
- **fine-tuned** — consulta apenas os modelos OpenAI fine-tuned.

## Estrutura do projeto

```
ask_llms/
  __init__.py
  config.py      # prompt do sistema, modos de execução e registro de modelos
  providers.py   # clientes de API e funções de consulta (OpenAI, Gemini, DeepSeek)
  pipeline.py    # processamento genérico do CSV (idempotente, salva a cada linha)
main.py          # ponto de entrada / CLI
data/            # CSVs de entrada e saída
```

## Instalação

Com o [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Ou com pip:

```bash
pip install -r requirements.txt
```

### Variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto:

```
OPENAI_API_KEY=sua_chave_aqui
GEMINI_API_KEY=sua_chave_aqui
DEEPSEEK_API_KEY=sua_chave_aqui

# Apenas necessárias no modo fine-tuned
FT_MODEL_NAME_GPT_4_1=ft:gpt-4.1:...
FT_MODEL_NAME_GPT_4_1_MINI=ft:gpt-4.1-mini:...
```

No modo `normal` só as três primeiras são necessárias; no modo `fine-tuned` só as chaves da OpenAI (`OPENAI_API_KEY` + as duas `FT_MODEL_NAME_*`) são usadas.

## Executar

```bash
# Modo normal (default) — todos os modelos base
python main.py

# Modo fine-tuning — apenas os modelos OpenAI fine-tuned
python main.py --mode fine-tuned
```

### Opções da CLI

| Flag            | Descrição                               | Default                                                                               |
| --------------- | --------------------------------------- | ------------------------------------------------------------------------------------- |
| `--mode`        | `normal` ou `fine-tuned`                | `normal`                                                                              |
| `--csv`         | Caminho do CSV a processar              | `data/perguntas_e_respostas.csv` (normal) / `data/normal_fined_tune.csv` (fine-tuned) |
| `--concurrency` | Nº de perguntas processadas em paralelo | `1` (normal) / `4` (fine-tuned)                                                       |

Exemplo customizando arquivo e concorrência:

```bash
python main.py --mode fine-tuned --csv data/outro_lote.csv --concurrency 6
```

O processamento é **idempotente**: se uma célula já tiver resposta, o modelo correspondente é pulado naquela linha — é seguro interromper e retomar a execução a qualquer momento.

## Estrutura do CSV

O CSV deve conter uma coluna `pergunta` (e opcionalmente `id`). As colunas dos modelos são criadas e preenchidas automaticamente conforme o modo escolhido.

**Modo normal:**

```
id,pergunta,gpt-5,gpt-5-mini,gpt-5-nano,gemini-2.5-flash,gemini-2.5-pro,deepseek-chat,deepseek-reasoner,gpt-4.1,gpt-4.1-mini
1,Há cultivares de macieira resistentes a todas as doenças da macieira?,,,,,,,
2,Há possibilidade de uso de drones de pulverização na cultura da macieira?,,,,,,,
```

**Modo fine-tuned:**

```
id,pergunta,gpt-4.1-ft,gpt-4.1-mini-ft
1,Há cultivares de macieira resistentes a todas as doenças da macieira?,,
2,Há possibilidade de uso de drones de pulverização na cultura da macieira?,,
```

## Modelos utilizados

**Modo normal**

- **OpenAI**: `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-4.1`, `gpt-4.1-mini`
- **Google**: `gemini-2.5-flash`, `gemini-2.5-pro`
- **DeepSeek**: `deepseek-chat`, `deepseek-reasoner`

**Modo fine-tuned**

- **OpenAI**: fine-tunes de `gpt-4.1` e `gpt-4.1-mini` (definidos via `FT_MODEL_NAME_GPT_4_1` e `FT_MODEL_NAME_GPT_4_1_MINI`)

## Adicionando um novo modelo ou provider

1. Se for um provider novo (ex.: outra API), adicione uma função `ask_*` em `ask_llms/providers.py` e registre-a no dicionário `PROVIDERS`.
2. Adicione o modelo à lista correspondente em `ask_llms/config.py` (`_normal_models()` ou `_fine_tuned_models()`), informando `column` (nome da coluna no CSV), `model_id` e o `provider` já registrado.

Nenhuma mudança em `pipeline.py` ou `main.py` é necessária — o pipeline é genérico sobre a lista de modelos configurada.

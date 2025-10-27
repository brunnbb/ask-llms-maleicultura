# Script para perguntar várias LLMs sobre maleicultura

Este script lê perguntas sobre maleicultura (cultivo de maças) de um arquivo CSV, envia cada pergunta para diversos modelos de linguagem (LLMs) e salva as respostas no mesmo CSV.

## Instalação 

### Instale as dependências necessárias:

   ```bash
   pip install requirements.txt
   ```

### Crie o arquivo .env na raiz do projeto com suas chaves de API:

   ```bash
    OPENAI_API_KEY=sua_chave_aqui
    GEMINI_API_KEY=sua_chave_aqui
    DEEPSEEK_API_KEY=sua_chave_aqui
   ```

## Executar
Estando no diretório do script execute:

   ```bash
   python main.py
   ```

## Estrutura do CSV 

O CSV deve conter a coluna de perguntas e os ids. As colunas dos modelos serão preenchidas automaticamente:

```csv
id,pergunta,gpt-5,gpt-5-mini,gpt-5-nano,gemini-2.5-flash,gemini-2.5-pro,deepseek-reasoner,deepseek-chat
1,Há cultivares de macieira resistentes a todas as doenças da macieira?,,,,,,,
2,Há possibilidade de uso de drones de pulverização na cultura da macieira?,,,,,,,,
```

## Modelos Utilizados

- **OpenAI**
  - `gpt-5`
  - `gpt-5-mini`
  - `gpt-5-nano`
- **Google**
  - `gemini-2.5-flash`
  - `gemini-2.5-pro`
- **DeepSeek**
  - `deepseek-chat`
  - `deepseek-reasoner`
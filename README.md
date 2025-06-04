# Projeto RAG com LangChain, Chroma e Ollama

Este projeto implementa um sistema de RAG (Retrieval-Augmented Generation) utilizando arquivos JSON como base de conhecimento. Os dados são vetorizados com LangChain e armazenados no ChromaDB. As respostas são geradas usando modelos locais via Ollama.

## Estrutura

```
├── ingest_data.py             # Script para carregar e indexar documentos no Chroma
├── query_rag.py               # Script para consultar o modelo com RAG
├── get_embedding_function.py  # Função de embedding com Ollama
├── json_loader.py             # Carregador e parser para arquivos JSON estruturados
├── requirements.txt           # Dependências do projeto
├── .gitignore                 # Arquivos/diretórios ignorados pelo Git
├── data_json/               # Diretório contendo os arquivos .json com os dados
└── chroma/                    # Diretório gerado automaticamente com a base vetorizada
```

## Requisitos

- Python 3.10+
- Ollama instalado e rodando localmente (`https://ollama.com`)
- Modelo `mxbai-embed-large` e `deepseek-r1:8b` disponíveis no Ollama

## Instalação

1. Clone o repositório:

```bash
git clone https://github.com/felipeduuartee/rag-oracle-icc
cd rag-oracle-icc
```

2. Crie um ambiente virtual (opcional, mas recomendado):

```bash
python3 -m venv .venv
source .venv/bin/activate  # no Windows: .venv\Scripts\activate
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

## Ingestão dos dados

Para criar ou atualizar a base de dados vetorizada:

```bash
python ingest_data.py
```

Para limpar a base e reconstruir do zero:

```bash
python ingest_data.py --reset
```

## Consulta

Para fazer uma pergunta ao sistema:

```bash
python query_rag.py "Qual pergunta devo fazer para descobrir a entrada correta?"
```

A resposta será gerada com base nos documentos mais relevantes e exibida no terminal, junto com as fontes utilizadas.

## Sobre os arquivos JSON

Este projeto trabalha com dois tipos principais de arquivos `.json`:
- Fases estruturadas com narrativa e desafio lógico
- Notícias com resumo, conteúdo técnico e implicações éticas

O script `json_loader.py` cuida automaticamente de identificar e extrair os campos relevantes.

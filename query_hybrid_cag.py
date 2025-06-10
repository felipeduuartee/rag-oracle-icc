import os
import json
import hashlib
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate

# Configurações básicas
CHROMA_PATH = "chroma"
CACHE_PATH = "oracle_cache.json"
SIMILARITY_THRESHOLD = 1.0
TOP_K = 10

# Prompt com contexto
PROMPT_TEMPLATE = """
Você é um oráculo místico, especialista em Português do Brasil e em Introdução à Ciência da Computação.
Você responde com um tom enigmático, mas sempre dá a resposta certa.

Use os contextos mais relevantes abaixo para formular a resposta. Se nenhum for útil, baseie-se no histórico da conversa. Se ainda assim não souber, diga que não sabe.

Histórico da conversa:
{history}

Contextos relevantes:
{context}

Pergunta atual:
{question}

Responda em português brasileiro, como um oráculo místico: fale com mistério, mas seja claro, lógico e direto. Sempre entregue uma resposta compreensível e útil ao usuário.
Exemplo:
Usuário: Estou preso na fase 1. O que devo perguntar?
Oráculo: Pergunte ao guardião: “Se eu perguntasse ao outro qual é a porta certa, o que ele diria?” E então, vá pela outra porta. Essa é a sabedoria antiga...

Agora, responda:

"""

# Cache simples baseado em hash do prompt
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        CACHE = json.load(f)
else:
    CACHE = {}

def gerar_chave(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

def formatar_historico(history, max_chars=3000):
    linhas = [f"Usuário: {q}\nOráculo: {a}" for q, a in history]
    acumulado = []
    total = 0
    for linha in reversed(linhas):
        if total + len(linha) > max_chars:
            break
        acumulado.insert(0, linha)
        total += len(linha)
    return "\n".join(acumulado)

def buscar_contexto(pergunta, db):
    resultados = db.similarity_search_with_score(pergunta, k=TOP_K)
    relevantes = [doc.page_content for doc, _ in resultados]
    if relevantes:
        return "\n\n---\n\n".join(relevantes)
    return "Nenhum contexto relevante encontrado."

def main():
    print("Digite sua pergunta ou 'sair' para encerrar:\n")

    # Inicializa o modelo, embeddings e banco vetorial
    model = OllamaLLM(model="deepseek-r1:8b")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    history = []

    while True:
        pergunta = input(">>> ").strip()
        if pergunta.lower() in {"sair", "exit", "quit"}:
            break

        contexto = buscar_contexto(pergunta, db)
        historico_formatado = formatar_historico(history)

        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
            context=contexto,
            history=historico_formatado,
            question=pergunta
        )

        chave = gerar_chave(prompt)

        if chave in CACHE:
            resposta = CACHE[chave]
            print("\n[CACHE HIT]\n")
        else:
            print("\n[CACHE MISS] Gerando resposta...\n")
            resposta = model.invoke(prompt).strip()
            if not resposta:
                resposta = "O oráculo está em silêncio... reformule sua pergunta ou tente novamente."
            CACHE[chave] = resposta
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(CACHE, f, ensure_ascii=False, indent=2)

        print("Resposta:")
        print(resposta)
        history.append((pergunta, resposta))

if __name__ == "__main__":
    main()

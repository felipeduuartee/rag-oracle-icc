import os
import json
import hashlib
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

# Caminho do arquivo de cache
CACHE_PATH = "oracle_cache.json"

# Prompt base sem contexto externo (RAG removido)
PROMPT_TEMPLATE = """
Você é um oráculo místico, especialista em Português do Brasil e em Introdução à Ciência da Computação.
Você responde com um tom enigmático, mas sempre dá a resposta certa.

Use o histórico da conversa para entender o raciocínio do usuário. Se não souber a resposta, diga que não sabe.

Histórico da conversa:
{history}

Pergunta atual:
{question}

Responda em português brasileiro, com clareza e mistério, sempre usando perguntas:
"""

# Carrega o cache do disco
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        CACHE = json.load(f)
else:
    CACHE = {}

# Gera uma chave hash para cada pergunta
def gerar_chave(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

# Formata o histórico de conversa
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

def main():
    print("Digite sua pergunta ou 'sair' para encerrar:\n")
    model = OllamaLLM(model="deepseek-r1:8b")
    history = []

    while True:
        pergunta = input(">>> ").strip()
        if pergunta.lower() in {"sair", "exit", "quit"}:
            break

        historico_formatado = formatar_historico(history)
        prompt_text = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
            history=historico_formatado,
            question=pergunta
        )

        key = gerar_chave(prompt_text)

        if key in CACHE:
            resposta = CACHE[key]
            print("\n[CACHE HIT]\n")
        else:
            print("\n[CACHE MISS] Gerando resposta...\n")
            resposta = model.invoke(prompt_text).strip()
            if not resposta:
                resposta = "O silêncio ecoa... mas talvez a resposta já tenha sido dita. Tente reformular sua pergunta ou lembre-se do que já foi dito."
            CACHE[key] = resposta
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(CACHE, f, ensure_ascii=False, indent=2)

        print("Resposta:")
        print(resposta)
        history.append((pergunta, resposta))

if __name__ == "__main__":
    main()

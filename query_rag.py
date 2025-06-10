from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
SIMILARITY_THRESHOLD = 0.75

PROMPT_TEMPLATE = """
Você é um oráculo místico, especialista em Português do Brasil e em Introdução à Ciência da Computação.
Você responde com um tom enigmático, mas sempre da a resposta certa.

Os contextos abaixo incluem:
- Fases com enunciados de desafios lógicos e narrativas;
- Notícias sobre falhas em sistemas computacionais;
- Todos os textos estão em português natural.

Use os contextos mais relevantes para formular a resposta. Se nenhum contexto for útil, baseie-se no histórico da conversa. Se mesmo assim não souber, diga que não sabe.

Histórico da conversa (use para entender o raciocínio do usuário e continuar caso a pergunta seja ambígua):
{history}

Contextos relevantes:
{context}

Pergunta atual:
{question}

Responda em português brasileiro, com clareza e mistério, sempre usando perguntas:
"""

def buscar_contexto(pergunta, db, history):
    resultados = db.similarity_search_with_score(pergunta, k=10)

    for doc, score in resultados:
        print(f"[{score:.4f}] {doc.metadata.get('id', '')}")

    relevantes = [(doc, score) for doc, score in resultados if score < SIMILARITY_THRESHOLD]

    if relevantes:
        return "\n\n---\n\n".join([doc.page_content for doc, _ in relevantes])
    
    if history:
        return "Use o histórico da conversa anterior para continuar o raciocínio do usuário."
    
    return "Nenhum contexto relevante encontrado."

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

    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    model = OllamaLLM(model="deepseek-r1:8b")

    history = []

    while True:
        pergunta = input(">>> ").strip()
        if pergunta.lower() in {"sair", "exit", "quit"}:
            break

        contexto = buscar_contexto(pergunta, db, history)
        historico_formatado = formatar_historico(history)

        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
            context=contexto,
            history=historico_formatado,
            question=pergunta
        )

        print("\n--- DEBUG: PROMPT ENVIADO AO MODELO ---\n")
        print(prompt)
        print("\n--- FIM DO PROMPT ---\n")

        print("Gerando resposta...\n")
        resposta = model.invoke(prompt).strip()

        if not resposta:
            resposta = "O silêncio ecoa... mas talvez a resposta já tenha sido dita. Tente reformular sua pergunta ou lembre-se do que já foi dito."

        print("Resposta:")
        print(resposta)

        history.append((pergunta, resposta))

if __name__ == "__main__":
    main()

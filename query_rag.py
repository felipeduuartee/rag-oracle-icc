import argparse
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
SIMILARITY_THRESHOLD = 0.5

PROMPT_TEMPLATE = """
Você é um assistente de IA especialista em Português do Brasil e em Introdução à Ciência da Computação.

Receba os contextos abaixo e use-os para responder à pergunta. Ignore contextos irrelevantes. Se nenhum contexto ajudar, admita que não sabe.

Contextos:

{context}

---

Pergunta: {question}

Responda em português brasileiro, de forma clara e objetiva:
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="Texto da pergunta.")
    args = parser.parse_args()
    rag_query(args.query_text)

# Executa o processo de RAG: busca, filtro, geração e resposta
def rag_query(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Busca os 10 documentos mais relevantes
    results = db.similarity_search_with_score(query_text, k=10)

    # Imprime os scores de similaridade
    for doc, score in results:
        print(f"[{score:.4f}] {doc.metadata.get('id', '')}")

    # Filtra apenas os documentos com score abaixo do limite
    filtered_results = [(doc, score) for doc, score in results if score < SIMILARITY_THRESHOLD]

    if not filtered_results:
        context_text = "Nenhum contexto relevante encontrado."
    else:
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in filtered_results])

    # Prepara o prompt com os contextos e a pergunta
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    print("\nGerando resposta...\n")
    model = OllamaLLM(model="deepseek-r1:8b")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id") for doc, _ in filtered_results]
    
    # Exibe a resposta final e as fontes utilizadas
    print("Resposta:")
    print(response_text)
    print("\nFontes:")
    print(sources)

if __name__ == "__main__":
    main()

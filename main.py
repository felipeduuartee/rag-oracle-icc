# Esse é o arquivo que eu estava testando
# Os outros eu mexi um pouco para testar
# nao coloquei a parada do oraculo nem cache/hash
# tem que rodar chroma run --path chroma-server pra rodar o banco chromaDB
# Passei os JSON pra pdf caso queira (diretorio PDF)
# Inclui algumas coisas no requirements também
# Alterei o modelo de embedding usado no banco



import argparse
import os
import re
import shutil

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain.schema import Document
from get_embedding_function import get_embedding_function
from pdf_loader import load_pdf
from ollama import chat
from langchain_core.documents import Document as LCDocument
from chromadb import Client
from chromadb.config import Settings
from chromadb import HttpClient

COLLECTION_NAME = "meu_banco"
CHROMA_SERVER_PATH = "chroma-server"

# Divide os documentos em pedaços menores (chunks)
def split_documents(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    return splitter.split_documents(documents)

# Junta os documentos recuperados em texto
def combine_docs(docs: list[LCDocument]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

# Consulta o modelo do Ollama
def ollama_llm(question, context):
    formatted_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    response = chat(
        model="deepseek-r1:14b",
        messages=[{"role": "user", "content": formatted_prompt}]
    )
    content = response['message']['content']
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

# Executa RAG
def rag_chain(question, retriever):
    retrieved_docs = retriever.invoke(question)
    context = combine_docs(retrieved_docs)
    return ollama_llm(question, context)

# Cria client para ChromaDB no modo servidor
def get_chroma_client():
    return HttpClient(host="localhost", port=8000)

# Função principal
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Resetar o banco de dados.")
    args = parser.parse_args()

    client = get_chroma_client()

    if args.reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(" Banco vetorial removido.")
        except Exception as e:
            print(f" Erro ao remover coleção: {e}")

    # Se ainda não existe a collection
    if COLLECTION_NAME not in [c.name for c in client.list_collections()]:
        print("🔍 Carregando documentos e criando base vetorial...")
        documents = load_pdf()
        chunks = split_documents(documents)

        Chroma.from_documents(
            documents=chunks,
            embedding=get_embedding_function(),
            collection_name=COLLECTION_NAME,
            client=client
        )
        print("Banco criado.")

    # Conecta ao banco vetorial
    db = Chroma(
        collection_name=COLLECTION_NAME,
        client=client,
        embedding_function=get_embedding_function()
    )
    retriever = db.as_retriever()

    # Loop de conversa
    print("🤖 Chatbot iniciado! Digite sua pergunta (ou 'sair' para encerrar):")
    while True:
        pergunta = input("❓ ")
        if pergunta.lower() in ["sair", "exit", "quit"]:
            break
        resposta = rag_chain(pergunta, retriever)
        print("🤖", resposta)
        print()

if __name__ == "__main__":
    main()

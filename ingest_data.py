import argparse
import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document
from get_embedding_function import get_embedding_function
from json_loader import load_json_documents
from pdf_loader import load_pdf

CHROMA_PATH = "chroma"
DATA_PATH = "data_json"

# Divide os documentos em pedaços menores (chunks)
def split_documents(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    return splitter.split_documents(documents)


def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model="deepseek-r1", messages=[{'role': 'user', 'content': formatted_prompt}])
    response_content = response['message']['content']
    # Remove content between <think> and </think> tags to remove thinking output
    final_answer = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL).strip()
    return final_answer


def rag_chain(question, text_splitter, vectorstore, retriever):
    retrieved_docs = retriever.invoke(question)
    formatted_content = combine_docs(retrieved_docs)
    return ollama_llm(question, formatted_content)
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Resetar o banco de dados.")
    args = parser.parse_args()

    if args.reset:
        print("Limpando o banco de dados existente...")
        clear_database()

    documents = load_pdf()
    chunks = split_documents(documents)
    #print(f"Total de chunks: {len(chunks)}")
    #print(f"Exemplo de chunk: {chunks[0].page_content if chunks else 'Nenhum'}")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_function(),
        persist_directory=CHROMA_PATH,        
    )
    
    
if __name__ == "__main__":
    main()

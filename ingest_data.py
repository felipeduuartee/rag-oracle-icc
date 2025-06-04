import argparse
import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document
from get_embedding_function import get_embedding_function
from json_loader import load_json_documents

CHROMA_PATH = "chroma"
DATA_PATH = "rag_ds_json"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Resetar o banco de dados.")
    args = parser.parse_args()

    if args.reset:
        print("Limpando o banco de dados existente...")
        clear_database()

    documents = load_json_documents(DATA_PATH)
    chunks = split_documents(documents)
    add_to_chroma(chunks)

# Divide os documentos em pedaços menores (chunks)
def split_documents(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_documents(documents)

# Gera um ID único para cada chunk, com base no nome do arquivo e índice
def calculate_chunks_ids(chunks: list[Document]):
    last_source = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        if source == last_source:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
            last_source = source

        chunk_id = f"{source}:{current_chunk_index}"
        chunk.metadata["id"] = chunk_id

    return chunks

# Adiciona os chunks ao banco Chroma, evitando duplicação por ID
def add_to_chroma(chunks: list[Document]):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    chunks_with_ids = calculate_chunks_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Documentos já existentes no banco: {len(existing_ids)}")

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"Adicionando {len(new_chunks)} novos documentos ao banco...")
        db.add_documents(new_chunks, ids=[chunk.metadata["id"] for chunk in new_chunks])
    else:
        print("Nenhum novo documento para adicionar.")

# Apaga a pasta do banco de dados Chroma
def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()

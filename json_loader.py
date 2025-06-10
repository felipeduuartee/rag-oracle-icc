import os
import json
from langchain.schema import Document

def load_json_documents(path: str) -> list[Document]:
    documents = []

    for filename in os.listdir(path):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(path, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Erro ao ler {filename}")
                continue

        if "fase" in data and "contexto_narrativo" in data:
            text = f"""
[Fase {data.get("fase")} – {data.get("titulo", "")}]

Contexto:
{data.get("contexto_narrativo", "").strip()}

Desafio:
{data.get("desafio_logico", "").strip()}

Pergunta principal:
{data.get("pergunta_principal", "").strip()}
"""
            doc_id = f"fase_{data['fase']:02d}"

        elif "summary" in data and "content" in data:
            conclusao = data.get("content", {}).get("conclusao", "")
            text = f"""
[Notícia: {data.get("title", data.get("titulo", ""))}]

Resumo:
{data.get("summary", data.get("resumo", ""))}

Conclusão:
{conclusao}
"""
            doc_id = data.get("id", filename.replace(".json", ""))

        else:
            print(f"Ignorando {filename}: formato não reconhecido.")
            continue

        documents.append(Document(
            page_content=text.strip(),
            metadata={"id": doc_id, "source": filename}
        ))

    return documents

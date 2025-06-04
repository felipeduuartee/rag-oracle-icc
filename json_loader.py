import os
import json
from langchain.schema import Document

# Função que carrega documentos do diretório de arquivos JSON
def load_json_documents(path: str) -> list[Document]:
    documents = []

    # Percorre todos os arquivos da pasta
    for filename in os.listdir(path):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(path, filename)
        
        # Tenta abrir e carregar o JSON
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Erro ao ler {filename}")
                continue

        # Arquivo do tipo "fase estruturada"
        if "fase" in data and "contexto_narrativo" in data:
            text = f"""
[Fase {data.get("fase")} – {data.get("titulo", "")}]

Contexto:
{data.get("contexto_narrativo", "")}

Desafio:
{data.get("desafio_logico", "")}

Pergunta:
{data.get("pergunta_principal", "")}
"""
            doc_id = f"fase_{data['fase']:02d}"

        # Arquivo do tipo "notícia"
        elif "summary" in data and "content" in data:
            conclusao = data.get("content", {}).get("conclusao", "")
            text = f"""
[Notícia: {data.get("title", "")}]

Resumo:
{data.get("summary", "")}

Conclusão:
{conclusao}
"""
            doc_id = data.get("id", filename.replace(".json", ""))

        # Arquivo com estrutura não reconhecida
        else:
            print(f"Ignorando {filename}: formato não reconhecido.")
            continue

        # Cria o objeto Document com o texto extraído e metadados
        documents.append(Document(
            page_content=text.strip(),
            metadata={"id": doc_id, "source": filename}
        ))

    return documents

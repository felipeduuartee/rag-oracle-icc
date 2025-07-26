import os
from langchain_community.document_loaders import PyMuPDFLoader

def load_pdf():
    documentos = []
    dir = "pdf"

    for file in os.listdir(dir):
        if file.lower().endswith(".pdf"):
            full_path = os.path.join(dir, file)
            try:
                loader = PyMuPDFLoader(full_path)
                data = loader.load()
                documentos.extend(data)
                print(f"✅ Carregado: {file} ({len(data)} páginas)")
            except Exception as e:
                print(f"❌ Erro ao carregar {file}: {e}")

    print(f"📄 Total de documentos carregados: {len(documentos)}")
    return documentos

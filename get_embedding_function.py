from langchain_ollama import OllamaEmbeddings

# função de embedding usada tanto para ingestão quanto consulta
def get_embedding_function():
    return OllamaEmbeddings(model="mxbai-embed-large")

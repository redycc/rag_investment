from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings


def get_embedding_function():
    
    #embeddings = OllamaEmbeddings(model="bge-m3")
    embeddings = OllamaEmbeddings(model="multilingual-e5-large")
    return embeddings

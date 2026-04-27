from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
import os

def openai_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")

def load_vectorstore(index_path = 'data/faiss_index'):
    embeddings=openai_embeddings()
    vector_store=FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization= True
    )
    
    return vector_store

def load_retriever(mode,vector_store,docs,top_k):
    if mode == "dense":
        return vector_store.as_retriever(search_kwargs={"k":top_k})
    
    if mode == "bm25":
        bm25= BM25Retriever.from_documents(docs)
        bm25.k=top_k
        return bm25
    if mode == "hybrid":
         dense= vector_store.as_retriever(search_kwargs={"k":top_k})
         bm25 = BM25Retriever.from_documents(docs)
         bm25.k = top_k
         
         hybrid = EnsembleRetriever(
             retrievers=[dense,bm25],
             weights=[0.5,0.5]
         )
         return hybrid
        
    else:
        raise ValueError(f"unknown mode {mode}")
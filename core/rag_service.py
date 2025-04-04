import os
from uuid import uuid4 
import traceback

import chromadb 
from chromadb.config import Settings
from openai import OpenAIError 
from core.llm_service import LLMService
from .mcp_service import ModelContextProvider

class RAGService:
    def __init__(
        self, 
        llm: LLMService, 
        vector_db_base : str = "./chromaDB",
        context_provider = None
    ):
        self.llm = llm
        self.context_provider = context_provider
        
        # provider 별 vector db 생성
        self.vector_db_path = os.path.join(vector_db_base, llm.provider)
        os.makedirs(self.vector_db_path, exist_ok = True)
        
        self.client = chromadb.Client(
            Settings(
                persist_directory = self.vector_db_path
            )
        )
        self.collection = self.client.get_or_create_collection("rag_knowledge")
    
    def query(self, question: str, user_id : str, threshold: float = 0.8):
        try:
            context = ""
            if self.context_provider:
                context = self.context_provider.getContext(user_id)
            full_prompt = f"{context}\n\nUser: {question}".strip()
            
            query_embedding = self.llm.getEmbedding(question)
            results = self.collection.query(
                query_embeddings = [query_embedding],
                n_results = 1
            )
            
            print("results", results)
            
            documents = results.get("documents", [])
            distances = results.get("distances", [])
            has_docs = documents and len(documents[0]) > 0
            if not has_docs or distances[0][0] > threshold:
                answer = self.llm.getAnswer(question)
                full_text = f"Q: {question}\nA: {answer}"
                answer_embedding = self.llm.getEmbedding(full_text)
                self.collection.add(
                    documents = [full_text],
                    embeddings = [answer_embedding],
                    ids = [str(uuid4())]
                )
                
                if self.context_provider:
                    self.context_provider.addContext(user_id, question, answer)
                
                return {
                    "source" : "llm",
                    "answer" : answer
                }
            else:
                return {
                    "source" : "db",
                    "answer" : results["documents"][0][0]
                }
        except Exception as e:
            traceback.print_exc();
            return {
                "error" : str(e),
                "provider" : self.llm.provider
            }
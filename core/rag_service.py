from uuid import uuid4 
import traceback

import chromadb 
from openai import OpenAIError 
from core.llm_service import LLMService

class RAGService:
    def __init__(self, llm: LLMService):
        self.llm = llm
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection("rag_knowledge")
    
    def query(self, question: str, threshold: float = 0.8):
        try:
            query_embedding = self.llm.getEmbedding(question)
            results = self.collection.query(
                query_embeddings = [query_embedding],
                n_results = 1
            )
            
            if not results["documents"] or results["distances"][0][0] > threshold:
                answer = self.llm.getAnswer(question)
                full_text = f"Q: {question}\nA: {answer}"
                answer_embedding = self.llm.getEmbedding(full_text)
                self.collection.add(
                    documents = [full_text],
                    embeddings = [answer_embedding],
                    ids = [str(uuid4())]
                )
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
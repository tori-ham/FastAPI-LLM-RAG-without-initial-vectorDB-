# import openai
import os
import requests 
from openai import OpenAI
from sentence_transformers import SentenceTransformer

class LLMService:
    def __init__(self, provider : str, apiKey : str):
        self.provider = provider
        self.apiKey = apiKey
        
        if provider == "openai":
            self.llmClient = OpenAI(api_key = apiKey)
        elif provider == "hf":
            self.apiUrl = os.getenv("HF_API_URL", "https://api-inference.huggingface.co/models/gpt2")
            
            self.embedModel = SentenceTransformer("all-MiniLM-L6-v2")
        elif provider == "cohere":
            import cohere
            self.llmClient = cohere.Client(apiKey)
        elif provider == "groq":
            self.apiUrl = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
        else:
            raise ValueError(f"Unsupported Provider : {provider}")    
    def getCompletion(self, prompt : str) -> str:
        openai_response = self.llmClient.completions.create(
            engine = "text-davinci-003",
            prompt = prompt,
            max_tokens = 500,
            temperature = 0.7
        )
        return openai_response.choices[0].text.strip()
    def getAnswer(self, prompt : str) -> str:
        if self.provider == "openai":
            response = self.llmClient.chat.completions.create(
                model = "gpt-3.5-turbo",
                messages=[
                    {
                        "role" : "user",
                        "content" : prompt
                    }
                ],
                temperature = 0.7
            )
            return response["choices"][0]["message"]["content"].strip()
        elif self.provider == "hf":
            headers = { "Authorization" : f"Bearer {self.apiKey}" }
            payload = { "inputs" : prompt }
            response = requests.post(
                self.apiUrl, 
                headers = headers,
                json = payload
            )
            return response.json()[0]["generated_text"]
        elif self.provider == "cohere":
            response = self.llmClient.chat(
                message = prompt, 
                model = "command-r"
            )
            return response.text
        elif self.provider == "groq":
            headers = {
                "Authorization" : f"Bearer {self.apiKey}",
                "Content-Type" : "application/json"
            }
            payload = {
                "model" : "mixtral-8x7b-32768",
                "messages" : [
                    {
                        "role" : "user",
                        "content" : prompt
                    }
                ],
                "temperature" : 0.7
            }
            response = requests.post(
                self.apiUrl,
                headers = headers,
                json = payload
            )
            return response.json()["choices"][0]["message"]["content"].strip()
    def getEmbedding(self, text : str) -> list:
        if self.provider == "openai":
            response = self.llmClient.embeddings.create(
                input = text,
                model = "text-embedding-ada-002"
            )
            return response["data"][0]["embedding"]
        elif self.provider == "hf":
            # embedding = self.embedModel.encode(text)
            return self.embedModel.encode(text).tolist()
        elif self.provider == "cohere":
            response = self.llmClient.embed(
                texts=[ text ],
                model="embed-english-v3.0",
                input_type="search_document"
            )
            return response.embeddings[0]
        elif self.provider == "groq":
            return self.embedModel.encode(text).tolist()
        else:
            raise NotImplementedError(f"Embedding not supported for provider: {self.provider}")
        
    
    
            
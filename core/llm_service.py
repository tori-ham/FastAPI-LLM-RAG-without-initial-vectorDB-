# import openai
from openai import OpenAI

class LLMService:
    def __init__(self, apiKey : str):
        self.llmClient = OpenAI(api_key = apiKey)
    def getCompletion(self, prompt : str) -> str:
        openai_response = self.llmClient.Completion.create(
            engine = "",
            prompt = prompt,
            max_tokens = 500,
            temperature = 0.7
        )
        return openai_response.choices[0].text.strip()
    def getAnswer(self, prompt : str) -> str:
        response = self.llmClient.ChatCompletion.create(
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
    def getEmbedding(self, text : str) -> list:
        response = self.llmClient.Embedding.create(
            input = text,
            model = "text-embedding-ada-002"
        )
        return response["data"][0]["embedding"]
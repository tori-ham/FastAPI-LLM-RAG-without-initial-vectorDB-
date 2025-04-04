import openai

class LLMService:
    def __init__(self, apiKey : str):
        openai.apiKey = apiKey
    def getCompletion(self, prompt : str) -> str:
        openai_response = openai.Completion.create(
            engine = "",
            prompt = prompt,
            max_tokens = 500,
            temperature = 0.7
        )
        return openai_response.choices[0].text.strip()
    def getAnswer(self, prompt : str) -> str:
        response = openai.ChatCompletion.create(
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
        response = openai.Embedding.create(
            input = text,
            model = "text-embedding-ada-002"
        )
        return response["data"][0]["embedding"]
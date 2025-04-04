import redis 

class ModelContextProvider:
    def __init__(
        self, 
        host="localhost", 
        port=6379, 
        db=0, 
        max_history=10
    ):
        self.client = redis.Redis(
            host = host,
            port = port,
            db = db,
            decode_responses = True
        )
        self.max_history = max_history
        
    def getContext(self, user_id : str) -> str:
        key = f"context:{user_id}"
        history = self.client.lrange(key, -self.max_history, -1)
        return "\n".join(history)
    
    def addContext(self, user_id : str, question : str, answer : str):
        key = f"context:{user_id}"
        entry = f"User: {question}\nAssistant: {answer}"
        self.client.rpush(key, entry)
        self.client.ltrim(key, -self.max_history, -1)
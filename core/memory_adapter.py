import os

import redis
from dotenv import load_dotenv
from typing import List, Optional

load_dotenv()

class RedisConversationMemory:
    def __init__(
        self, 
        userId : str,
        redisClient : Optional[redis.Redis] = None,
        maxHistory : int = 5,
        keyPrefix : str = "context"
    ) : 
        self.user_id = userId 
        self.key = f"{keyPrefix}:{userId}"
        self.max_history = maxHistory
        self.redis = redisClient or redis.Redis(
            host = os.environ.get("REDIS_HOST"),
            port = os.environ.get("REDIS_PORT"),
            db = os.environ.get("REDIS_DB"),
            decode_responses = True
        )
    
    def loadMemoryVariables(self) -> dict:
        history = self.redis.lrange(self.key, -self.max_history, -1)
        return {
            "history" : "\n".join(history)
        }
        
    def saveContext(self, inputs : dict, outputs : dict):
        question = inputs.get("question") or inputs.get("input")
        answer = outputs.get("answer") or outputs.get("output")
        entry = f"User: {question}\nAnswer: {answer}"
        self.redis.rpush(self.key, entry)
        self.redis.ltrim(self.key, -self.max_history, -1)
    
    def clearContext(self):
        self.redis.delete(self.key)
    
    def getAllHistory(self) -> list:
        return self.redis.lrange(self.key, 0, -1)
    
    def saveSummary(self, summary : str):
        self.redis.set(f"{self.key}:summary", summary)
    
    def loadSummary(self) -> Optional[str]:
        return self.redis.get(f"{self.key}:summary")
from fastapi import APIRouter, Depends, Path
# from pydantic import BaseModel
from core.rag_service import RAGService
from core.llm_service import LLMService
from app.dependencies import getLLMService
from app.config import AppSettings
from app.dependencies import getSettings
from core.memory_adapter import RedisConversationMemory

from models.QuestionRequest import QuestionRequest

router = APIRouter()
# mcpService = ModelContextProvider()

@router.post("/ask/{provider}")
async def askQuestion(
    settings : AppSettings = Depends(getSettings),
    provider : str = Path(..., regex="^(openai|hf|cohere|groq)$"),
    request : QuestionRequest = ...,
):
    llm = getLLMService(provider, settings)
    
    memory = RedisConversationMemory(request.userId)
    rag = RAGService(
        llm, 
        userId = request.userId,
        vector_db_base=settings.vector_db_base,
        context_provider = memory,
        memory = memory
    )
    rag.summarizeNStoreHistory()
    result = rag.query(
        request.question,
        user_id = request.userId)
    return result
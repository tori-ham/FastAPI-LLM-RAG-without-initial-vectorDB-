from fastapi import APIRouter, Depends, Path
# from pydantic import BaseModel
from core.rag_service import RAGService
from core.llm_service import LLMService
from app.dependencies import getLLMService
from app.config import AppSettings
from app.dependencies import getSettings

from models.QuestionRequest import QuestionRequest

router = APIRouter()

@router.post("/ask/{provider}")
async def askQuestion(
    settings : AppSettings = Depends(getSettings),
    provider : str = Path(..., regex="^(openai|hf|cohere|groq)$"),
    request : QuestionRequest = ...,
):
    llm = getLLMService(provider, settings)
    rag = RAGService(llm, vector_db_base=settings.vector_db_base)
    result = rag.query(request.question)
    return result
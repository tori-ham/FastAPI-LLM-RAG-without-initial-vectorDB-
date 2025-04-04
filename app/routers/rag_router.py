from fastapi import APIRouter, Depends
# from pydantic import BaseModel
from core.rag_service import RAGService
from core.llm_service import LLMService
from app.dependencies import getLLMService

from models.QuestionRequest import QuestionRequest

router = APIRouter()

@router.post("/ask")
async def askQuestion(
    request : QuestionRequest,
    llm : LLMService = Depends(getLLMService)
):
    rag = RAGService(llm)
    result = rag.query(request.question)
    return result
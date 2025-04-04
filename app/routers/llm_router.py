import traceback

from fastapi import APIRouter, Depends

from app.dependencies import verifyAPIKey
from core.llm_service import LLMService

router = APIRouter()

@router.post("/completion")
async def generateCompletion(
    prompt : str,
    apiKey = Depends(verifyAPIKey)
) : 
    llmService = LLMService( apiKey = apiKey.openai_api_key )
    result = llmService.getCompletion(prompt)
    return {
        "prompt" : prompt,
        "completion" : result
    }
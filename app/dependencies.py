from fastapi import Depends, HTTPException
from .config import AppSettings

import os 
from core.llm_service import LLMService

def getSettings():
    return AppSettings()

def verifyAPIKey(
    settings : AppSettings = Depends(getSettings)
) :
    if not settings.openai_api_key:
        raise HTTPException(
            status_code = 400,
            detail = "API Key is missing"
        )
    return settings

def getLLMService(
    settings : AppSettings = Depends(getSettings)
):
    if not settings.openai_api_key:
        raise HTTPException(
            status_code = 400,
            detail = "API Key is missing"
        )
    return LLMService(apiKey=settings.openai_api_key)
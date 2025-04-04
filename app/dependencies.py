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
    provider : str,
    settings : AppSettings
) -> LLMService:
    keyMap = {
        "openai" : settings.openai_api_key,
        "hf" : settings.hf_api_key,
        "cohere" : settings.cohere_api_key,
        "groq" : settings.groq_api_key
    }
    
    if provider not in keyMap:
        raise ValueError("지원하지 않는 provider입니다.")
    return LLMService(provider = provider, apiKey=keyMap[provider])
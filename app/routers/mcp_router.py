from fastapi import APIRouter
from core.mcp_service import MCPService

router = APIRouter()

@router.get("/status")
async def mcpStatus():
    mcp_service = MCPService()
    return mcp_service.getStatus()

@router.post("context")
async def setContext(contextData : dict):
    mcpService = MCPService()
    mcpService.setContext(contextData)
    return {
        "message" : "Context Updated"
    }
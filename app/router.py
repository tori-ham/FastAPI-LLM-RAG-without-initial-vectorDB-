from fastapi import APIRouter

from app.routers import llm_router, mcp_router, rag_router

router = APIRouter()
# router.include_router( llm_router.router, prefix = "/api/llm", tags = [ "LLM" ] )
# router.include_router( mcp_router.router, prefix = "/api/mcp", tags = [ "MCP" ] )
router.include_router( rag_router.router, prefix = "/api/rag", tags = [ "RAG" ] )
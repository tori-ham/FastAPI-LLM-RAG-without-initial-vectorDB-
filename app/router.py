from fastapi import APIRouter

from app.routers import rag_router

router = APIRouter()
router.include_router( rag_router.router, prefix = "/api", tags = [ "RAG" ] )
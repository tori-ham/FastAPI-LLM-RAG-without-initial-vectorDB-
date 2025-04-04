from fastapi import FastAPI
from app.config import AppSettings

# from app import router
from app.router import router

def create_app() -> FastAPI:
    appSettings = AppSettings()
    app = FastAPI(
        title = appSettings.app_title,
        description = appSettings.description,
        version = appSettings.app_version
    )
    app.include_router(router)
    
    return app

app = create_app()

if __name__ == "__main__" :
    import uvicorn 
    uvicorn.run(
        "app.app:app", 
        host="0.0.0.0"
    )
from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="Codebase Intelligence Platform")

app.include_router(router)


@app.get("/")
def root():
    return {
        "status": "running",
        "message": "Codebase Intelligence Platform is running"
    }
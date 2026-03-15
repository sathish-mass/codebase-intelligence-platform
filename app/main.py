from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="AI Codebase Assistant")

app.include_router(router)

@app.get("/")
def health():
    return {"status": "AI Codebase Assistant running"}
from fastapi import APIRouter

router = APIRouter()

@router.get("/test")
def test_api():
    return {"message": "API working"}
"""信息抽取课程补充实验的交互 API。"""

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1/information-extraction", tags=["information-extraction"])


class ExtractionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


@router.post("/analyze")
def analyze(payload: ExtractionRequest, request: Request) -> dict:
    return request.app.state.information_extraction.analyze(payload.text)

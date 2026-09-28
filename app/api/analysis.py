from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field, field_validator

router = APIRouter(prefix="/api/v1")


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    tasks: list[Literal["segmentation", "ner"]] = Field(
        default_factory=lambda: ["segmentation", "ner"], min_length=1
    )
    run_id: str = "latest"
    segmentation_model: Literal["hmm1-smooth", "longest", "bigram", "hmm2-smooth", "bilstm-crf"] = (
        "hmm1-smooth"
    )

    @field_validator("tasks")
    @classmethod
    def unique_tasks(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("tasks 不得重复")
        return value


@router.post("/analyze")
def analyze(body: AnalyzeRequest, request: Request) -> dict:
    return request.app.state.nlp.analyze(
        body.text, body.tasks, body.run_id, body.segmentation_model
    )

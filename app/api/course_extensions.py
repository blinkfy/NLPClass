"""课程序列标注与词性标注交互 API；不接触 official test。"""

from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1/course")


class SequenceRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    model: Literal["perceptron", "crf"]


class POSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)


@router.get("/resources")
def resources(request: Request) -> dict:
    return request.app.state.course.resources()


@router.post("/sequence")
def sequence(body: SequenceRequest, request: Request) -> dict:
    return request.app.state.course.sequence(body.text, body.model)


@router.post("/pos")
def pos(body: POSRequest, request: Request) -> dict:
    return request.app.state.course.pos_tag(body.text)

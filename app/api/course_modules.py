"""Independent course modules; no frozen experiment writes or training routes."""

import json
from threading import Event
from typing import Literal

import anyio

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1/course")


class DependencyInput(BaseModel):
    text: str = Field(min_length=1, max_length=300)


class DeepInput(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    task: Literal["chat", "summary", "extract", "boundary"] = "chat"
    max_new_tokens: int = Field(default=48, ge=1, le=64)
    sampling: bool = False


@router.get("/modules/resources")
def resources(request: Request):
    return request.app.state.course_neural.resources()


@router.post("/dependency/parse")
def dependency(body: DependencyInput, request: Request):
    return request.app.state.course_neural.dependency(body.text)


@router.get("/dependency/evaluation")
def dependency_evaluation(request: Request):
    return request.app.state.course_neural.dependency_evaluation()


@router.post("/deep/{kind}")
def deep(kind: Literal["bilstm", "bert", "llm"], body: DeepInput, request: Request):
    return request.app.state.course_neural.infer(kind, body.model_dump())


@router.post("/deep/llm/stream", response_class=StreamingResponse,
             responses={200: {"content": {"text/event-stream": {}}}})
async def llm_stream(body: DeepInput, request: Request):
    stop = Event()
    iterator = request.app.state.course_neural.stream_llm(body.model_dump(), stop)
    def advance():
        return next(iterator, None)
    async def events():
        try:
            while not await request.is_disconnected():
                event = await anyio.to_thread.run_sync(advance)
                if event is None:
                    break
                yield f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False, allow_nan=False)}\n\n"
                if event["event"] in ("done", "error"):
                    break
        finally:
            stop.set()
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(iterator.close)
    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

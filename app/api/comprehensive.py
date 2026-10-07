"""Experiment four application routes; old model APIs stay unchanged."""

from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1/comprehensive", tags=["comprehensive application"])


class TextInput(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


class OptionalInput(TextInput):
    kind: Literal["information_extraction", "dependency", "summary"]
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=1)


class CustomDocument(TextInput):
    title: str = Field(default="", max_length=100)


class CustomInput(BaseModel):
    documents: list[CustomDocument] = Field(min_length=3, max_length=30)
    k: int = Field(default=3, ge=2, le=29)


class LinkInput(BaseModel):
    document_ids: list[str] = Field(default_factory=list, max_length=10)


@router.get("/status")
def status(request: Request):
    return request.app.state.comprehensive.status()


@router.post("/analyze")
def analyze(body: TextInput, request: Request):
    return request.app.state.comprehensive.analyze(body.text)


@router.post("/optional")
def optional(body: OptionalInput, request: Request):
    return request.app.state.comprehensive.optional(body.text, body.kind, body.start, body.end)


@router.get("/course")
def course(request: Request):
    return request.app.state.comprehensive.course_collection()


@router.post("/custom")
def custom(body: CustomInput, request: Request):
    return request.app.state.comprehensive.custom_collection([doc.model_dump() for doc in body.documents], body.k)


@router.get("/collections/{collection_id}/documents/{document_id}")
def profile(collection_id: str, document_id: str, request: Request, analyze: bool = False):
    return request.app.state.comprehensive.profile(collection_id, document_id, analyze=analyze)


@router.post("/collections/{collection_id}/link")
def link(collection_id: str, body: LinkInput, request: Request):
    return request.app.state.comprehensive.link(collection_id, body.document_ids)


@router.post("/system-tests")
def system_tests(request: Request):
    return request.app.state.comprehensive.system_tests()


class SamplingInput(BaseModel):
    scope_ids: list[str] | None = Field(default=None, max_length=600)
    limit: int = Field(default=10, ge=1, le=10)


class TermsInput(BaseModel):
    document_ids: list[str] = Field(default_factory=list, max_length=600)


@router.get("/collections/{collection_id}/workspace")
def workspace(collection_id: str, request: Request):
    return request.app.state.comprehensive.workspace_snapshot(collection_id)


@router.post("/collections/{collection_id}/sampling-plan")
def sampling(collection_id: str, body: SamplingInput, request: Request):
    return request.app.state.comprehensive.sampling_plan(collection_id, body.scope_ids, body.limit)


@router.post("/collections/{collection_id}/terms")
def terms(collection_id: str, body: TermsInput, request: Request):
    return request.app.state.comprehensive.selection_terms(collection_id, body.document_ids)


@router.get("/cases")
def cases():
    from nlplab.comprehensive.workspace import replay_cases
    return {"cases": replay_cases(), "model_outputs_predefined": False}


@router.post("/classification-index/build")
def build_classification_index(request: Request):
    return request.app.state.comprehensive.build_classification_index()

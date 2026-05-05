from __future__ import annotations

import io
from typing import Optional

from anthropic import AsyncAnthropic
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from ..config import settings
from ..models import (
    ATSRequest,
    ATSResult,
    CoverLetterRequest,
    ExportRequest,
    OptimizeRequest,
    ResumeData,
)
from ..services import ats_scorer, exporter, resume_optimizer, resume_parser

router = APIRouter(prefix="/api/resume", tags=["resume"])


def _get_client() -> AsyncAnthropic:
    if not settings.anthropic_api_key:
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY not configured. Add it to your .env file.",
        )
    return AsyncAnthropic(api_key=settings.anthropic_api_key)


@router.post("/upload", response_model=ResumeData)
async def upload_resume(file: UploadFile = File(...)):
    content_type = file.content_type or ""
    file_bytes = await file.read()

    try:
        if "pdf" in content_type or file.filename.endswith(".pdf"):
            return await resume_parser.parse_pdf(file_bytes)
        elif "word" in content_type or "docx" in content_type or file.filename.endswith(".docx"):
            return await resume_parser.parse_docx(file_bytes)
        else:
            text = file_bytes.decode("utf-8", errors="replace")
            return await resume_parser.parse_text(text)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse resume: {e}")


@router.post("/paste", response_model=ResumeData)
async def paste_resume(body: dict):
    text = body.get("text", "")
    if not text.strip():
        raise HTTPException(status_code=422, detail="No resume text provided")
    return await resume_parser.parse_text(text)


@router.post("/ats-score", response_model=ATSResult)
async def ats_score(request: ATSRequest):
    client = None
    if request.use_claude and settings.anthropic_api_key:
        client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    return await ats_scorer.compute_ats_score(
        request.resume_text,
        request.job_description,
        request.use_claude,
        client,
    )


@router.post("/optimize")
async def optimize_resume_sse(request: OptimizeRequest):
    client = _get_client()

    async def generator():
        try:
            async for chunk in resume_optimizer.optimize_resume_streaming(
                client,
                request.resume_data,
                request.target_role,
                request.job_description,
            ):
                yield {"data": chunk, "event": "chunk"}
            yield {"data": "[DONE]", "event": "done"}
        except Exception as e:
            yield {"data": f"[ERROR] {e}", "event": "error"}

    return EventSourceResponse(generator())


@router.post("/optimize-full")
async def optimize_resume_full(request: OptimizeRequest):
    client = _get_client()
    try:
        text = await resume_optimizer.optimize_resume_full(
            client,
            request.resume_data,
            request.target_role,
            request.job_description,
        )
        return {"optimized_resume": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export")
async def export_resume(request: ExportRequest):
    try:
        docx_bytes = exporter.generate_docx(request.optimized_text, request.contact)
        return StreamingResponse(
            io.BytesIO(docx_bytes),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": "attachment; filename=resume_optimized.docx"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")


@router.post("/cover-letter")
async def cover_letter_sse(request: CoverLetterRequest):
    client = _get_client()

    async def generator():
        try:
            async for chunk in resume_optimizer.generate_cover_letter_streaming(client, request):
                yield {"data": chunk, "event": "chunk"}
            yield {"data": "[DONE]", "event": "done"}
        except Exception as e:
            yield {"data": f"[ERROR] {e}", "event": "error"}

    return EventSourceResponse(generator())

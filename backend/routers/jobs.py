from __future__ import annotations

from typing import Optional

from fastapi import APIRouter

from ..config import settings
from ..models import JobSearchResult
from ..services.ats_scorer import score_keywords
from ..services.job_searcher import search_jobs

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("/search", response_model=JobSearchResult)
async def search_jobs_endpoint(
    role: str = "cybersecurity",
    location: str = "remote",
    salary_min: int = 0,
    page: int = 1,
    resume_text: Optional[str] = None,
):
    return await search_jobs(
        role=role,
        location=location,
        salary_min=salary_min,
        page=page,
        resume_text=resume_text,
        adzuna_app_id=settings.adzuna_app_id,
        adzuna_app_key=settings.adzuna_app_key,
    )


@router.post("/match-score")
async def match_score(body: dict):
    resume_text = body.get("resume_text", "")
    jd_text = body.get("job_description", "")
    if not resume_text or not jd_text:
        return {"score": 0, "matched": [], "missing": []}

    result = score_keywords(resume_text, jd_text)
    return {
        "score": result["keyword_score"],
        "matched": sorted(result["matched"])[:30],
        "missing": sorted(result["missing"])[:30],
    }

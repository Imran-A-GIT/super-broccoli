from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

import httpx

from ..models import JobListing, JobSearchResult
from .ats_scorer import score_keywords

logger = logging.getLogger(__name__)

ROLE_SEARCH_TERMS: dict[str, list[str]] = {
    "tech_sales": [
        "solutions engineer",
        "sales engineer",
        "technical account manager",
        "presales consultant",
    ],
    "product_owner": [
        "product owner",
        "product manager",
        "scrum master",
        "agile product manager",
    ],
    "cybersecurity": [
        "security analyst",
        "SOC analyst",
        "information security analyst",
        "cybersecurity analyst",
    ],
    "data_analyst": [
        "data analyst",
        "business intelligence analyst",
        "reporting analyst",
        "BI analyst",
    ],
}

MUSE_CATEGORY_MAP: dict[str, str] = {
    "tech_sales": "Sales",
    "product_owner": "Product",
    "cybersecurity": "IT",
    "data_analyst": "Data Science",
}

_ADZUNA_BASE = "https://api.adzuna.com/v1/api/jobs/us/search"
_MUSE_BASE = "https://www.themuse.com/api/public/jobs"


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text).strip()


async def search_adzuna(
    client: httpx.AsyncClient,
    app_id: str,
    app_key: str,
    keywords: str,
    location: str,
    salary_min: int,
    page: int = 1,
    results_per_page: int = 20,
) -> list[JobListing]:
    if not app_id or not app_key:
        return []

    params = {
        "app_id": app_id,
        "app_key": app_key,
        "what": keywords,
        "where": location or "united states",
        "salary_min": salary_min if salary_min > 0 else "",
        "results_per_page": results_per_page,
        "content-type": "application/json",
    }
    params = {k: v for k, v in params.items() if v != ""}

    try:
        resp = await client.get(f"{_ADZUNA_BASE}/{page}", params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Adzuna search failed: %s", exc)
        return []

    jobs: list[JobListing] = []
    for item in data.get("results", []):
        jobs.append(
            JobListing(
                title=item.get("title", ""),
                company=item.get("company", {}).get("display_name", "Unknown"),
                location=item.get("location", {}).get("display_name"),
                salary_min=item.get("salary_min"),
                salary_max=item.get("salary_max"),
                url=item.get("redirect_url", ""),
                description=_strip_html(item.get("description", ""))[:500],
                source="adzuna",
            )
        )
    return jobs


async def search_muse(
    client: httpx.AsyncClient,
    keywords: str,
    category: str,
    location: str,
    page: int = 0,
) -> list[JobListing]:
    params = {
        "page": page,
        "category": category,
    }
    if location and location.lower() not in ("remote", "us", "usa", "united states"):
        params["location"] = location

    try:
        resp = await client.get(_MUSE_BASE, params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("The Muse search failed: %s", exc)
        return []

    # Filter by keyword relevance (Muse doesn't support keyword search in free API)
    kw_lower = keywords.lower().split()
    jobs: list[JobListing] = []
    for item in data.get("results", []):
        title = item.get("name", "")
        if not any(kw in title.lower() for kw in kw_lower):
            continue
        locs = item.get("locations", [])
        loc_name = locs[0]["name"] if locs else location
        url = item.get("refs", {}).get("landing_page", "")
        contents = _strip_html(item.get("contents", ""))[:500]
        jobs.append(
            JobListing(
                title=title,
                company=item.get("company", {}).get("name", "Unknown"),
                location=loc_name,
                salary_min=None,
                salary_max=None,
                url=url,
                description=contents,
                source="muse",
            )
        )
    return jobs


def _deduplicate(jobs: list[JobListing]) -> list[JobListing]:
    seen: set[tuple[str, str]] = set()
    unique: list[JobListing] = []
    for job in jobs:
        key = (job.title.lower()[:40], job.company.lower()[:30])
        if key not in seen:
            seen.add(key)
            unique.append(job)
    return unique


def _sort_by_salary(jobs: list[JobListing]) -> list[JobListing]:
    return sorted(
        jobs,
        key=lambda j: (j.salary_min is None, -(j.salary_min or 0)),
    )


async def search_jobs(
    role: str,
    location: str,
    salary_min: int,
    page: int,
    resume_text: Optional[str],
    adzuna_app_id: str,
    adzuna_app_key: str,
) -> JobSearchResult:
    terms = ROLE_SEARCH_TERMS.get(role, ["technology professional"])
    primary_term = terms[0]
    muse_category = MUSE_CATEGORY_MAP.get(role, "IT")

    async with httpx.AsyncClient(timeout=15.0) as client:
        adzuna_results, muse_results = await asyncio.gather(
            search_adzuna(
                client,
                adzuna_app_id,
                adzuna_app_key,
                primary_term,
                location,
                salary_min,
                page=page,
            ),
            search_muse(client, primary_term, muse_category, location, page=page - 1),
        )

    combined = _deduplicate(adzuna_results + muse_results)
    sorted_jobs = _sort_by_salary(combined)

    if resume_text and sorted_jobs:
        for job in sorted_jobs:
            if job.description:
                try:
                    result = score_keywords(resume_text, job.description)
                    job.match_score = result["keyword_score"]
                except Exception:
                    pass

    return JobSearchResult(
        jobs=sorted_jobs,
        total=len(sorted_jobs),
        page=page,
    )

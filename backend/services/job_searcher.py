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
    "custom": [
        "technology professional",
        "technical specialist",
        "IT professional",
    ],
}

MUSE_CATEGORY_MAP: dict[str, str] = {
    "tech_sales": "Sales",
    "product_owner": "Product",
    "cybersecurity": "IT",
    "data_analyst": "Data Science",
}

REMOTIVE_CATEGORY_MAP: dict[str, str] = {
    "tech_sales": "sales",
    "product_owner": "product",
    "cybersecurity": "devops-sysadmin",
    "data_analyst": "data",
}

JOBICY_TAG_MAP: dict[str, str] = {
    "tech_sales": "sales-engineer",
    "product_owner": "product-manager",
    "cybersecurity": "security",
    "data_analyst": "data-analytics",
    "custom": "",
}

_ADZUNA_US_BASE = "https://api.adzuna.com/v1/api/jobs/us/search"
_ADZUNA_CA_BASE = "https://api.adzuna.com/v1/api/jobs/ca/search"
_MUSE_BASE = "https://www.themuse.com/api/public/jobs"
_REMOTIVE_BASE = "https://remotive.com/api/remote-jobs"
_JOBICY_BASE = "https://jobicy.com/api/v2/remote-jobs"

_REMOTE_LOCATIONS = frozenset(
    {"remote", "us", "usa", "united states", "anywhere", "worldwide", ""}
)
_CANADA_CITIES = frozenset({
    "toronto", "vancouver", "montreal", "calgary", "ottawa", "edmonton",
    "winnipeg", "hamilton", "quebec", "victoria", "canada",
})

# Words to exclude from title-relevance filter
_FILTER_STOP = frozenset({"of", "and", "the", "or", "a", "an", "in", "at", "for"})


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text).strip()


def _parse_salary_str(s: str) -> tuple[Optional[float], Optional[float]]:
    """Parse salary strings like '$80k-$120k' or '80000-120000' → (min, max)."""
    if not s:
        return None, None
    nums: list[float] = []
    for m in re.finditer(r"[\d,]+\.?\d*k?", s.lower()):
        part = m.group().replace(",", "")
        if part.endswith("k"):
            try:
                nums.append(float(part[:-1]) * 1000)
            except ValueError:
                pass
        else:
            try:
                n = float(part)
                if n >= 10000:
                    nums.append(n)
            except ValueError:
                pass
    return (nums[0] if nums else None, nums[1] if len(nums) > 1 else None)


def _adzuna_base(location: str) -> str:
    loc = (location or "").lower()
    if any(c in loc for c in _CANADA_CITIES):
        return _ADZUNA_CA_BASE
    return _ADZUNA_US_BASE


async def search_adzuna(
    client: httpx.AsyncClient,
    app_id: str,
    app_key: str,
    keywords: str,
    location: str,
    salary_min: int,
    page: int = 1,
    results_per_page: int = 50,
) -> list[JobListing]:
    if not app_id or not app_key:
        return []

    params = {
        "app_id": app_id,
        "app_key": app_key,
        "what": keywords,
        "where": location or "united states",
        "results_per_page": results_per_page,
        "content-type": "application/json",
    }
    if salary_min > 0:
        params["salary_min"] = salary_min
    params = {k: v for k, v in params.items() if v != ""}

    try:
        resp = await client.get(f"{_adzuna_base(location)}/{page}", params=params)
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
    params: dict = {"page": page, "category": category}
    if location and location.lower() not in ("remote", "us", "usa", "united states"):
        params["location"] = location

    try:
        resp = await client.get(_MUSE_BASE, params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("The Muse search failed: %s", exc)
        return []

    kw_parts = keywords.lower().split()
    primary_kw = kw_parts[0]
    full_phrase = keywords.lower()
    jobs: list[JobListing] = []
    for item in data.get("results", []):
        title = item.get("name", "")
        title_lower = title.lower()
        if primary_kw not in title_lower and full_phrase not in title_lower:
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


async def search_remotive(
    client: httpx.AsyncClient,
    keywords: str,
    role: str,
    limit: int = 50,
) -> list[JobListing]:
    category = REMOTIVE_CATEGORY_MAP.get(role, "")
    params: dict = {"search": keywords, "limit": limit}
    if category:
        params["category"] = category

    try:
        resp = await client.get(_REMOTIVE_BASE, params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Remotive search failed: %s", exc)
        return []

    jobs: list[JobListing] = []
    for item in data.get("jobs", []):
        salary_min, salary_max = _parse_salary_str(item.get("salary", ""))
        jobs.append(
            JobListing(
                title=item.get("title", ""),
                company=item.get("company_name", "Unknown"),
                location=item.get("candidate_required_location") or "Remote",
                salary_min=salary_min,
                salary_max=salary_max,
                url=item.get("url", ""),
                description=_strip_html(item.get("description", ""))[:500],
                source="remotive",
            )
        )
    return jobs


async def search_jobicy(
    client: httpx.AsyncClient,
    role: str,
    limit: int = 50,
) -> list[JobListing]:
    """Jobicy — free remote job board with good salary data, no API key needed."""
    tag = JOBICY_TAG_MAP.get(role, "")
    params: dict = {"count": limit}
    if tag:
        params["tag"] = tag

    try:
        resp = await client.get(_JOBICY_BASE, params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Jobicy search failed: %s", exc)
        return []

    jobs: list[JobListing] = []
    for item in data.get("jobs", []):
        salary_min_val = None
        salary_max_val = None
        salary_type = (item.get("salaryType") or "").lower()
        for field, target in [("salaryMin", "min"), ("salaryMax", "max")]:
            raw = item.get(field)
            if raw:
                try:
                    val = float(str(raw).replace(",", "").replace("$", ""))
                    if "hour" in salary_type:
                        val *= 2080  # Convert hourly → annual
                    if target == "min":
                        salary_min_val = val
                    else:
                        salary_max_val = val
                except (ValueError, TypeError):
                    pass

        jobs.append(
            JobListing(
                title=item.get("jobTitle", ""),
                company=item.get("companyName", "Unknown"),
                location=item.get("jobGeo") or "Remote",
                salary_min=salary_min_val,
                salary_max=salary_max_val,
                url=item.get("url", ""),
                description=_strip_html(item.get("jobDescription", ""))[:500],
                source="jobicy",
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
    muse_category = MUSE_CATEGORY_MAP.get(role, "IT")
    include_remote = (location or "").lower().strip() in _REMOTE_LOCATIONS

    # Build title-relevance filter from ALL search terms for this role
    filter_kw: set[str] = set()
    for term in terms:
        filter_kw.update(term.lower().split())
    filter_kw -= _FILTER_STOP

    async with httpx.AsyncClient(timeout=20.0) as client:
        tasks: list = []

        # Adzuna: search first 3 role terms in parallel (50 results each)
        for term in terms[:3]:
            tasks.append(
                search_adzuna(client, adzuna_app_id, adzuna_app_key, term,
                              location, salary_min, page=1, results_per_page=50)
            )

        # The Muse: 3 pages in parallel (no salary, but good volume)
        for pg in range(3):
            tasks.append(search_muse(client, terms[0], muse_category, location, page=pg))

        # Remote-only boards — only when user is searching remote/US
        if include_remote:
            tasks.append(search_remotive(client, terms[0], role, limit=50))
            tasks.append(search_jobicy(client, role, limit=50))

        results = await asyncio.gather(*tasks, return_exceptions=True)

    all_jobs: list[JobListing] = []
    for r in results:
        if isinstance(r, list):
            all_jobs.extend(r)

    # Title relevance filter: drop jobs where none of the role keywords appear in title
    if filter_kw:
        all_jobs = [j for j in all_jobs if any(kw in j.title.lower() for kw in filter_kw)]

    combined = _deduplicate(all_jobs)
    sorted_jobs = _sort_by_salary(combined)

    # Filter out jobs with known salary below the user's minimum
    if salary_min > 0:
        sorted_jobs = [
            j for j in sorted_jobs
            if j.salary_min is None or j.salary_min >= salary_min
        ]

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

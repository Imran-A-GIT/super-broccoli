from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict


# --- Resume schemas ---

class ContactInfo(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    location: Optional[str] = None


class WorkExperience(BaseModel):
    title: str
    company: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    bullets: list[str] = []


class Education(BaseModel):
    degree: str
    institution: str
    year: Optional[str] = None


class ResumeData(BaseModel):
    contact: ContactInfo = ContactInfo()
    summary: Optional[str] = None
    experience: list[WorkExperience] = []
    skills: list[str] = []
    education: list[Education] = []
    certifications: list[str] = []
    raw_text: str = ""


# --- ATS schemas ---

class ATSRequest(BaseModel):
    resume_text: str
    job_description: str
    use_claude: bool = True


class ATSResult(BaseModel):
    score: int
    matched_keywords: list[str]
    missing_keywords: list[str]
    section_scores: dict[str, int]
    recommendations: list[str]
    semantic_matches: list[dict] = []


# --- Optimize schemas ---

class TargetRole(str, Enum):
    TECH_SALES = "tech_sales"
    PRODUCT_OWNER = "product_owner"
    CYBERSECURITY = "cybersecurity"
    DATA_ANALYST = "data_analyst"
    CUSTOM = "custom"


class OptimizeRequest(BaseModel):
    resume_data: ResumeData
    target_role: TargetRole
    job_description: Optional[str] = None
    emphasis: Optional[list[str]] = None


class ExportRequest(BaseModel):
    optimized_text: str
    contact: ContactInfo


class CoverLetterRequest(BaseModel):
    resume_data: ResumeData
    job_title: str
    company: str
    job_description: str
    target_role: TargetRole


# --- Job schemas ---

class JobListing(BaseModel):
    title: str
    company: str
    location: Optional[str] = None
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    url: str
    description: str = ""
    source: str
    match_score: Optional[int] = None


class JobSearchResult(BaseModel):
    jobs: list[JobListing]
    total: int
    page: int


# --- Application schemas ---

class ApplicationCreate(BaseModel):
    job_title: str
    company: str
    url: Optional[str] = None
    status: str = "Applied"
    applied_date: Optional[datetime] = None
    follow_up_date: Optional[datetime] = None
    notes: Optional[str] = None
    location: Optional[str] = None
    salary_range: Optional[str] = None
    target_role: Optional[str] = None
    ats_score: Optional[float] = None


class ApplicationUpdate(BaseModel):
    job_title: Optional[str] = None
    company: Optional[str] = None
    url: Optional[str] = None
    status: Optional[str] = None
    applied_date: Optional[datetime] = None
    follow_up_date: Optional[datetime] = None
    notes: Optional[str] = None
    ats_score: Optional[float] = None


class ApplicationOut(ApplicationCreate):
    id: int
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)


class DashboardStats(BaseModel):
    total: int
    by_status: dict[str, int]
    response_rate: float
    interviews_scheduled: int
    offers: int
    follow_ups_due: list[ApplicationOut]

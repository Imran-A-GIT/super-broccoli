from __future__ import annotations

import io
import re
from typing import Optional

import pdfplumber
from docx import Document

from ..models import ContactInfo, Education, ResumeData, WorkExperience

# Section header detection
_SECTION_RE = re.compile(
    r"^(SUMMARY|OBJECTIVE|PROFESSIONAL SUMMARY|PROFILE|"
    r"EXPERIENCE|WORK EXPERIENCE|EMPLOYMENT HISTORY|PROFESSIONAL EXPERIENCE|"
    r"SKILLS|TECHNICAL SKILLS|CORE COMPETENCIES|KEY SKILLS|"
    r"EDUCATION|ACADEMIC BACKGROUND|"
    r"CERTIFICATIONS?|LICENSES? & CERTIFICATIONS?|"
    r"PROJECTS?|ACHIEVEMENTS?|ACCOMPLISHMENTS?|"
    r"AWARDS?|PUBLICATIONS?|LANGUAGES?|INTERESTS?|HOBBIES?)\s*$",
    re.IGNORECASE,
)

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"(\+?1?\s?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}")
_LINKEDIN_RE = re.compile(r"linkedin\.com/in/[\w\-]+", re.IGNORECASE)
_DATE_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{4}|\d{4}\s*[-–]\s*(?:\d{4}|Present|Current|Now)\b",
    re.IGNORECASE,
)
_DEGREE_RE = re.compile(
    r"\b(Bachelor|Master|MBA|PhD|Ph\.D|Associate|B\.?S\.?|B\.?A\.?|M\.?S\.?|M\.?A\.?|"
    r"B\.?Sc|M\.?Sc|Doctor|Doctorate)\b",
    re.IGNORECASE,
)
_CERT_RE = re.compile(
    r"\b(Certified|Certificate|Certification|CompTIA|CISSP|CISM|CISA|CEH|"
    r"AWS|Azure|GCP|Google Cloud|PMP|ITIL|Scrum|PMI|CAPM|"
    r"Security\+|Network\+|A\+|CySA\+|CCNA|CCNP|MCSA|MCSE)\b",
    re.IGNORECASE,
)


async def parse_pdf(file_bytes: bytes) -> ResumeData:
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        pages_text = []
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
    full_text = "\n".join(pages_text)
    if not full_text.strip():
        raise ValueError("Could not extract text from PDF. Try a text-based PDF or paste your resume.")
    return _parse_text(full_text)


async def parse_docx(file_bytes: bytes) -> ResumeData:
    doc = Document(io.BytesIO(file_bytes))
    lines = []
    for para in doc.paragraphs:
        if para.text.strip():
            lines.append(para.text)
    full_text = "\n".join(lines)
    return _parse_text(full_text)


async def parse_text(raw_text: str) -> ResumeData:
    return _parse_text(raw_text)


def _parse_text(text: str) -> ResumeData:
    lines = [l.rstrip() for l in text.splitlines()]
    contact = _extract_contact(lines, text)
    sections = _split_sections(lines)
    summary = _extract_summary(sections)
    experience = _extract_experience(sections)
    skills = _extract_skills(sections)
    education = _extract_education(sections)
    certifications = _extract_certifications(sections, text)

    return ResumeData(
        contact=contact,
        summary=summary,
        experience=experience,
        skills=skills,
        education=education,
        certifications=certifications,
        raw_text=text,
    )


def _extract_contact(lines: list[str], full_text: str) -> ContactInfo:
    email_match = _EMAIL_RE.search(full_text)
    phone_match = _PHONE_RE.search(full_text)
    linkedin_match = _LINKEDIN_RE.search(full_text)

    email = email_match.group() if email_match else None
    phone = phone_match.group() if phone_match else None
    linkedin = linkedin_match.group() if linkedin_match else None

    # Name is typically the first non-empty line that isn't an email/phone/URL
    name = None
    for line in lines[:8]:
        stripped = line.strip()
        if (
            stripped
            and not _EMAIL_RE.search(stripped)
            and not _PHONE_RE.search(stripped)
            and not _SECTION_RE.match(stripped)
            and len(stripped) > 2
            and len(stripped) < 60
        ):
            name = stripped
            break

    # Location: look for City, ST pattern
    location = None
    loc_re = re.compile(r"\b[A-Z][a-zA-Z\s]+,\s*[A-Z]{2}\b")
    loc_match = loc_re.search(full_text)
    if loc_match:
        location = loc_match.group()

    return ContactInfo(name=name, email=email, phone=phone, linkedin=linkedin, location=location)


def _split_sections(lines: list[str]) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current_section = "HEADER"
    sections[current_section] = []

    for line in lines:
        if _SECTION_RE.match(line.strip()):
            current_section = line.strip().upper()
            sections[current_section] = []
        else:
            sections[current_section].append(line)

    return sections


def _extract_summary(sections: dict[str, list[str]]) -> Optional[str]:
    for key in sections:
        if any(w in key for w in ("SUMMARY", "OBJECTIVE", "PROFILE")):
            text = "\n".join(l for l in sections[key] if l.strip())
            return text.strip() or None
    return None


def _extract_experience(sections: dict[str, list[str]]) -> list[WorkExperience]:
    exp_lines: list[str] = []
    for key in sections:
        if any(w in key for w in ("EXPERIENCE", "EMPLOYMENT", "WORK")):
            exp_lines = sections[key]
            break

    if not exp_lines:
        return []

    jobs: list[WorkExperience] = []
    current_title = ""
    current_company = ""
    current_start = ""
    current_end = ""
    current_bullets: list[str] = []

    def _flush():
        if current_title or current_company:
            jobs.append(
                WorkExperience(
                    title=current_title.strip(),
                    company=current_company.strip(),
                    start_date=current_start or None,
                    end_date=current_end or None,
                    bullets=[b for b in current_bullets if b],
                )
            )

    for line in exp_lines:
        stripped = line.strip()
        if not stripped:
            continue

        date_match = _DATE_RE.search(stripped)
        bullet_match = re.match(r"^[•\-\*•]\s+(.+)", stripped)

        if bullet_match:
            current_bullets.append(bullet_match.group(1).strip())
        elif date_match and len(stripped) < 120:
            # Likely a job header line — save previous job first
            _flush()
            current_bullets = []
            current_start = ""
            current_end = ""

            # Try to parse "Title at Company | Date"
            date_str = date_match.group()
            rest = stripped.replace(date_str, "").strip().strip("|").strip()

            # Split on " at " or " - " for title/company
            if " at " in rest:
                parts = rest.split(" at ", 1)
                current_title = parts[0].strip()
                current_company = parts[1].strip()
            elif " | " in rest:
                parts = rest.split(" | ", 1)
                current_title = parts[0].strip()
                current_company = parts[1].strip()
            elif " – " in rest or " - " in rest:
                sep = " – " if " – " in rest else " - "
                parts = rest.split(sep, 1)
                current_title = parts[0].strip()
                current_company = parts[1].strip() if len(parts) > 1 else ""
            else:
                current_title = rest
                current_company = ""

            # Parse date range
            dates = _DATE_RE.findall(stripped)
            if len(dates) >= 2:
                current_start = dates[0]
                current_end = dates[1]
            elif len(dates) == 1:
                current_start = dates[0]
        elif not current_title and len(stripped) < 80 and not stripped.startswith("http"):
            # Possibly the title line without a date on the same line
            _flush()
            current_title = stripped
            current_company = ""
            current_bullets = []
        elif current_title and not current_company and len(stripped) < 80:
            current_company = stripped

    _flush()
    return jobs


def _extract_skills(sections: dict[str, list[str]]) -> list[str]:
    for key in sections:
        if any(w in key for w in ("SKILL", "COMPETENC", "EXPERTISE", "TECHNICAL")):
            raw = "\n".join(sections[key])
            # Split on commas, semicolons, pipes, newlines, bullets
            items = re.split(r"[,;|\n•\-\*]", raw)
            skills = [s.strip() for s in items if s.strip() and len(s.strip()) > 1]
            return skills[:60]
    return []


def _extract_education(sections: dict[str, list[str]]) -> list[Education]:
    for key in sections:
        if "EDUCATION" in key or "ACADEMIC" in key:
            edu_list = []
            lines = [l.strip() for l in sections[key] if l.strip()]
            i = 0
            while i < len(lines):
                line = lines[i]
                if _DEGREE_RE.search(line):
                    degree = line
                    institution = lines[i + 1] if i + 1 < len(lines) else ""
                    year_match = re.search(r"\b(19|20)\d{2}\b", line + " " + institution)
                    year = year_match.group() if year_match else None
                    edu_list.append(
                        Education(
                            degree=degree.strip(),
                            institution=institution.strip(),
                            year=year,
                        )
                    )
                    i += 2
                else:
                    i += 1
            return edu_list
    return []


def _extract_certifications(sections: dict[str, list[str]], full_text: str) -> list[str]:
    for key in sections:
        if "CERT" in key or "LICENSE" in key:
            lines = [l.strip() for l in sections[key] if l.strip()]
            certs = []
            for line in lines:
                clean = re.sub(r"^[•\-\*]\s*", "", line)
                if clean:
                    certs.append(clean)
            return certs

    # Fallback: find cert-like lines anywhere in the text
    certs = []
    for line in full_text.splitlines():
        stripped = line.strip()
        if _CERT_RE.search(stripped) and len(stripped) < 100:
            clean = re.sub(r"^[•\-\*]\s*", "", stripped)
            if clean and clean not in certs:
                certs.append(clean)
    return certs[:10]

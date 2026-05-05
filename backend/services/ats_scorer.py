from __future__ import annotations

import json
import re
from typing import Optional

from anthropic import AsyncAnthropic

from ..models import ATSResult

_STOP_WORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of",
        "with", "is", "are", "was", "were", "be", "been", "have", "has", "had",
        "will", "strong", "excellent", "experience", "knowledge", "skills",
        "ability", "demonstrated", "proven", "working", "familiarity",
        "proficiency", "required", "preferred", "must", "should", "may",
        "years", "year", "plus", "including", "such", "as", "etc", "minimum",
        "least", "ability", "us", "we", "you", "your", "our", "their", "this",
        "that", "these", "those", "not", "no", "do", "does", "did", "can",
        "could", "would", "should", "shall", "might", "need", "want",
    }
)

# Multi-word tech terms to detect as single units
_TECH_BIGRAMS = {
    "machine learning", "deep learning", "data analysis", "data analytics",
    "incident response", "network monitoring", "network security", "cloud computing",
    "project management", "product management", "agile methodology", "scrum master",
    "business intelligence", "data warehouse", "sql server", "power bi",
    "security operations", "vulnerability management", "penetration testing",
    "stakeholder management", "change management", "risk management",
    "disaster recovery", "business continuity", "zero trust", "identity management",
    "access control", "patch management", "threat detection", "security awareness",
    "technical support", "help desk", "service desk", "it operations",
    "infrastructure management", "system administration", "network administration",
    "cloud migration", "digital transformation", "devops", "ci/cd",
    "python programming", "data visualization", "artificial intelligence",
    "customer success", "account management", "solution selling",
}

_TOKEN_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+#.\-]{1,}\b")

_ACTION_VERBS = {
    "led", "managed", "developed", "designed", "implemented", "built",
    "created", "established", "achieved", "improved", "reduced", "increased",
    "delivered", "launched", "spearheaded", "orchestrated", "architected",
    "streamlined", "optimized", "automated", "deployed", "maintained",
    "monitored", "resolved", "coordinated", "collaborated", "supported",
}


def _extract_keywords(text: str) -> list[str]:
    lower = text.lower()
    # First extract bigrams
    found: set[str] = set()
    for bigram in _TECH_BIGRAMS:
        if bigram in lower:
            found.add(bigram)

    # Then single tokens
    tokens = _TOKEN_RE.findall(lower)
    for tok in tokens:
        if tok not in _STOP_WORDS and len(tok) > 2 and tok not in _ACTION_VERBS:
            found.add(tok)

    return list(found)


def score_keywords(resume_text: str, jd_text: str) -> dict:
    jd_keywords = set(_extract_keywords(jd_text))
    resume_keywords = set(_extract_keywords(resume_text))

    matched = jd_keywords & resume_keywords
    missing = jd_keywords - resume_keywords

    keyword_score = int(len(matched) / max(len(jd_keywords), 1) * 100)
    return {
        "keyword_score": keyword_score,
        "matched": matched,
        "missing": missing,
        "jd_keywords": jd_keywords,
    }


def score_sections(resume_text: str) -> dict[str, int]:
    lower = resume_text.lower()
    scores: dict[str, int] = {}

    # Summary: exists and has content
    has_summary = bool(
        re.search(r"\b(summary|objective|profile)\b", lower)
        and len(re.findall(r"\.\s", resume_text)) >= 2
    )
    scores["summary"] = 90 if has_summary else 30

    # Experience: has jobs with bullets
    bullet_count = len(re.findall(r"[•\-\*]\s+\w", resume_text))
    if bullet_count >= 10:
        scores["experience"] = 95
    elif bullet_count >= 5:
        scores["experience"] = 75
    elif bullet_count >= 2:
        scores["experience"] = 50
    else:
        scores["experience"] = 20

    # Skills: count skill-like items
    skills_section = re.search(r"skill[s\w]*.*?(?=\n[A-Z]{3,}|\Z)", lower, re.DOTALL)
    skill_count = 0
    if skills_section:
        skill_count = len(re.split(r"[,;|\n]", skills_section.group()))
    scores["skills"] = min(100, 60 + skill_count * 4)

    # Education: has a degree
    has_edu = bool(
        re.search(
            r"\b(bachelor|master|mba|phd|associate|b\.?s|b\.?a|m\.?s|degree)\b",
            lower,
        )
    )
    scores["education"] = 90 if has_edu else 40

    # Quantification: count numbers and % in bullet lines
    bullets = re.findall(r"[•\-\*]\s+.+", resume_text)
    bullets_with_numbers = sum(1 for b in bullets if re.search(r"\d+", b))
    quant_pct = int(bullets_with_numbers / max(len(bullets), 1) * 100) if bullets else 0
    scores["quantification"] = quant_pct

    return scores


async def semantic_match_claude(
    client: AsyncAnthropic,
    resume_text: str,
    missing_keywords: list[str],
) -> list[dict]:
    if not missing_keywords or not client:
        return []

    top_missing = missing_keywords[:20]
    prompt = (
        f"Resume excerpt:\n{resume_text[:2000]}\n\n"
        f"Missing keywords from job description: {top_missing}\n\n"
        "For each missing keyword, check if the resume contains a semantically equivalent term "
        "(e.g. 'network monitoring' ≈ 'infrastructure monitoring'). "
        "Return ONLY a JSON array: "
        '[{"resume_term": "...", "jd_term": "...", "confidence": 0.0-1.0}] '
        "Include only matches with confidence >= 0.7. Return [] if none."
    )

    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system="You are an ATS expert. Respond only with valid JSON, no markdown.",
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        raw = response.content[0].text.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        return json.loads(raw)
    except Exception:
        return []


def _build_recommendations(
    missing_keywords: list[str],
    section_scores: dict[str, int],
    semantic_matches: list[dict],
) -> list[str]:
    recs = []

    semantic_jd_terms = {m["jd_term"] for m in semantic_matches}
    truly_missing = [k for k in missing_keywords if k not in semantic_jd_terms][:10]

    if truly_missing:
        recs.append(f"Add these missing keywords to your resume: {', '.join(truly_missing[:8])}")

    if section_scores.get("summary", 100) < 60:
        recs.append("Add a professional summary (2-3 sentences) at the top of your resume")

    if section_scores.get("experience", 100) < 60:
        recs.append("Add more bullet points under each role (aim for 3-5 bullets per position)")

    if section_scores.get("skills", 100) < 70:
        recs.append("Expand your Skills section with at least 8-12 relevant technical skills")

    if section_scores.get("education", 100) < 60:
        recs.append("Include your education (degree, institution, year) clearly in your resume")

    if section_scores.get("quantification", 100) < 40:
        recs.append(
            "Quantify your achievements — add numbers, percentages, or dollar amounts to at least 50% of your bullets"
        )

    if semantic_matches:
        examples = semantic_matches[:2]
        for m in examples:
            recs.append(
                f"Consider adding '{m['jd_term']}' — your '{m['resume_term']}' covers this but ATS may not match it"
            )

    return recs


async def compute_ats_score(
    resume_text: str,
    jd_text: str,
    use_claude: bool,
    client: Optional[AsyncAnthropic],
) -> ATSResult:
    kw_result = score_keywords(resume_text, jd_text)
    section_scores = score_sections(resume_text)

    semantic_matches: list[dict] = []
    if use_claude and client and kw_result["missing"]:
        try:
            semantic_matches = await semantic_match_claude(
                client, resume_text, list(kw_result["missing"])
            )
        except Exception:
            semantic_matches = []

    # Adjust keyword score upward for semantic matches with high confidence
    semantic_bonus = sum(1 for m in semantic_matches if m.get("confidence", 0) >= 0.85)
    adjusted_keyword_score = min(
        100,
        kw_result["keyword_score"] + int(semantic_bonus / max(len(kw_result["jd_keywords"]), 1) * 100),
    )

    avg_section = int(sum(section_scores.values()) / max(len(section_scores), 1))
    final_score = int(0.6 * adjusted_keyword_score + 0.4 * avg_section)

    matched = sorted(kw_result["matched"])
    missing = sorted(kw_result["missing"])

    recs = _build_recommendations(missing, section_scores, semantic_matches)

    return ATSResult(
        score=final_score,
        matched_keywords=matched[:30],
        missing_keywords=missing[:30],
        section_scores=section_scores,
        recommendations=recs,
        semantic_matches=semantic_matches,
    )

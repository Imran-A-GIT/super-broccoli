from __future__ import annotations

from datetime import date
from typing import AsyncIterator, Optional

from anthropic import AsyncAnthropic

from ..models import CoverLetterRequest, OptimizeRequest, ResumeData

BASE_SYSTEM_PROMPT = """You are an expert resume writer specializing in career transitions from \
IT Operations roles. You have deep knowledge of ATS systems, hiring manager expectations, \
and how to translate technical operational experience into compelling narratives for \
adjacent technology roles.

Your rewrites:
1. Preserve factual accuracy — never invent metrics or experiences
2. Strengthen quantification — use [X] placeholders where specific numbers are missing
3. Use strong action verbs: spearheaded, architected, orchestrated, delivered, drove, engineered
4. Follow modern resume best practices: tight bullets under 2 lines, no first-person pronouns
5. Ensure ATS compatibility: standard section headers, no tables or text boxes
6. Output clean, structured text using the exact format requested"""

ROLE_PROFILES: dict[str, str] = {
    "tech_sales": """
ROLE: Technical Sales / Solutions Engineer / Technical Account Manager

OBJECTIVE: Transform IT Operations experience into a compelling technical sales narrative.
Emphasize the candidate's ability to bridge technical complexity and business value.

KEY THEMES TO EMPHASIZE:
- Client relationship management and stakeholder communication
- Technical advisory and solution design experience
- Translating infrastructure knowledge into business outcomes
- SLA management and service delivery (reframe as customer success)
- Problem-solving under pressure (reframe as overcoming objections, building trust)
- Cross-functional collaboration (reframe as account team coordination)
- Revenue and cost impact of IT decisions
- Demos, POCs, technical presentations (even internal ones count)

POWER PHRASES:
"drove adoption of", "partnered with stakeholders to", "delivered measurable ROI",
"technical trusted advisor", "solution-oriented", "customer-centric approach",
"accelerated time-to-value", "executive-level presentation", "presales support"

KEYWORDS TO WEAVE IN NATURALLY:
pipeline, revenue impact, solution selling, technical demonstrations, RFP/RFI,
value proposition, customer success, account management, upsell, cross-sell,
technical discovery, proof of concept, competitive differentiation

REFRAMING GUIDE:
- "Managed servers/infrastructure" → "Delivered reliable technical platforms supporting $[X]M in business operations"
- "Responded to incidents" → "Owned customer-facing technical escalations, maintaining [X]% SLA adherence"
- "Monitored networks" → "Proactively identified and resolved technical risks before customer impact"
- "Wrote runbooks" → "Developed technical documentation enabling faster onboarding and knowledge transfer"
""",
    "product_owner": """
ROLE: Product Owner / Product Manager / Scrum Master

OBJECTIVE: Reframe IT Operations experience as product and delivery leadership.
Emphasize structured decision-making, prioritization, and cross-team coordination.

KEY THEMES TO EMPHASIZE:
- Backlog management and prioritization (IT tickets = backlog items)
- Sprint/release planning and delivery cadence
- Stakeholder management and requirements gathering
- KPIs, metrics, and data-driven decisions
- Roadmap ownership and strategic planning
- User story thinking (internal users = end users)
- Change management and feature rollouts

POWER PHRASES:
"defined requirements for", "collaborated with engineering teams to",
"managed product backlog", "drove cross-functional alignment",
"launched initiative resulting in", "established OKRs for",
"facilitated sprint ceremonies", "gathered and prioritized stakeholder feedback"

KEYWORDS TO WEAVE IN NATURALLY:
backlog, sprint, user stories, acceptance criteria, MVP, roadmap, OKR, KPI,
velocity, retrospective, stakeholder, requirements, epics, scrum, agile,
product lifecycle, feature prioritization, release management

REFRAMING GUIDE:
- "Managed IT projects" → "Owned end-to-end product delivery with defined acceptance criteria and sprint planning"
- "Worked with vendors" → "Managed third-party integrations and partner roadmap alignment"
- "Created IT policies" → "Defined operational standards and drove organization-wide adoption"
- "Reduced downtime" → "Improved platform reliability from [X]% to [Y]%, impacting [N] users"
""",
    "cybersecurity": """
ROLE: Security Analyst / SOC Analyst / Information Security Engineer

OBJECTIVE: Elevate IT Operations experience into a security-focused narrative.
Surface any security-adjacent work (firewalls, access control, compliance, incident handling).

KEY THEMES TO EMPHASIZE:
- Security monitoring and threat detection (SIEM, log analysis)
- Incident response and forensics lifecycle
- Vulnerability management and patch cadence
- Access control and identity management (IAM, PAM)
- Compliance frameworks (SOC2, ISO 27001, NIST CSF, PCI-DSS, HIPAA)
- Network security (firewalls, VPN, IDS/IPS, network segmentation)
- Security hardening and baseline configuration
- Risk assessment and mitigation planning

POWER PHRASES:
"implemented security controls", "reduced attack surface by",
"led incident response for", "achieved compliance with",
"threat intelligence-driven", "zero-trust approach",
"hardened systems against", "monitored for indicators of compromise"

KEYWORDS TO WEAVE IN NATURALLY:
SIEM, SOC, threat hunting, vulnerability assessment, IAM, PAM, firewall,
IDS/IPS, DLP, endpoint security, patch management, NIST CSF, CIS benchmarks,
security hardening, CVE, IOC, MITRE ATT&CK, incident response, forensics

REFRAMING GUIDE:
- "Managed firewall rules" → "Engineered network security controls reducing unauthorized access attempts by [X]%"
- "Handled IT incidents" → "Led incident response lifecycle from detection through containment and remediation"
- "Set up monitoring" → "Deployed and tuned SIEM dashboards enabling real-time threat detection"
- "Managed user access" → "Implemented least-privilege IAM policies across [N] systems and [N] user accounts"
""",
    "data_analyst": """
ROLE: Data Analyst / Business Intelligence Analyst / Reporting Analyst

OBJECTIVE: Surface quantitative and analytical aspects of IT Operations work.
Reframe metrics, monitoring, and reporting as data analysis competencies.

KEY THEMES TO EMPHASIZE:
- Data collection, cleaning, and analysis
- Dashboard and report creation for stakeholders
- SQL and database management
- Metrics definition and KPI tracking
- Business insights from operational data
- Automation of reporting workflows
- Trend analysis and capacity planning (reframe as forecasting)

POWER PHRASES:
"analyzed [X] data points to identify", "built dashboard tracking",
"automated reporting reducing manual effort by [X]%",
"translated data insights into actionable recommendations",
"defined metrics framework for", "SQL queries supporting [N] stakeholders",
"visualized operational trends", "data-driven decision making"

KEYWORDS TO WEAVE IN NATURALLY:
SQL, Python, Excel, Tableau, Power BI, Looker, ETL, KPI, metrics,
reporting, dashboard, trend analysis, forecasting, data quality,
data governance, business intelligence, ad-hoc analysis, data pipeline

REFRAMING GUIDE:
- "Monitored system performance" → "Analyzed performance metrics across [N] systems, identifying trends and anomalies"
- "Created IT reports" → "Designed automated reporting dashboards tracking [X] KPIs for executive stakeholder review"
- "Managed asset inventory" → "Maintained data integrity across asset database of [N] records with [X]% accuracy"
- "Capacity planning" → "Forecasted infrastructure demand using 12-month historical trend analysis"
""",
}

CUSTOM_SYSTEM_PROMPT = """CUSTOM ROLE OPTIMIZATION MODE

A specific job description has been provided. Your primary guide is that job description.

Your approach:
1. Analyze the job description to extract required/preferred skills, responsibilities, and keywords
2. Map the candidate's IT Operations experience to those specific requirements
3. Use the employer's exact terminology where the experience genuinely matches
4. Surface any experience that transfers to this specific role, even if indirect
5. Prioritize and lead with the most relevant experience for this posting
6. Naturally weave in keywords from the job description throughout the resume
7. The job description drives every rewriting decision — tailor, don't generalize"""

ROLE_LABELS: dict[str, str] = {
    "tech_sales": "Technical Sales / Solutions Engineer",
    "product_owner": "Product Owner / Product Manager",
    "cybersecurity": "Security Analyst / Cybersecurity Engineer",
    "data_analyst": "Data Analyst / Business Intelligence Analyst",
    "custom": "the target role",
}


def _build_system_blocks(target_role: str) -> list[dict]:
    if target_role == "custom":
        return [{"type": "text", "text": BASE_SYSTEM_PROMPT + "\n\n" + CUSTOM_SYSTEM_PROMPT}]
    return [
        {"type": "text", "text": BASE_SYSTEM_PROMPT},
        {
            "type": "text",
            "text": ROLE_PROFILES[target_role],
            "cache_control": {"type": "ephemeral"},
        },
    ]


def _build_resume_user_prompt(resume_data: ResumeData, job_description: Optional[str]) -> str:
    c = resume_data.contact
    contact_line = " | ".join(filter(None, [c.email, c.phone, c.location, c.linkedin]))

    exp_text = ""
    for exp in resume_data.experience:
        dates = f"{exp.start_date or ''} – {exp.end_date or ''}"
        bullets = "\n".join(f"- {b}" for b in exp.bullets) if exp.bullets else "- [No bullets found]"
        exp_text += f"\n### {exp.title} | {exp.company} | {dates}\n{bullets}\n"

    edu_text = "\n".join(
        f"{e.degree} — {e.institution}" + (f" ({e.year})" if e.year else "")
        for e in resume_data.education
    ) or "Not specified"

    cert_text = "\n".join(f"- {c}" for c in resume_data.certifications) or "None listed"

    jd_block = ""
    if job_description:
        jd_block = f"\n--- TARGET JOB DESCRIPTION ---\n{job_description[:2000]}\n--- END JOB DESCRIPTION ---\n"

    return f"""Here is the candidate's resume:

--- CANDIDATE RESUME ---
NAME: {c.name or 'Not specified'}
CONTACT: {contact_line}

CURRENT SUMMARY:
{resume_data.summary or 'None provided'}

WORK EXPERIENCE:
{exp_text}

SKILLS: {', '.join(resume_data.skills) or 'Not listed'}

EDUCATION:
{edu_text}

CERTIFICATIONS:
{cert_text}
--- END RESUME ---
{jd_block}

Your task:
1. Rewrite this resume to position the candidate for the target role defined in your system instructions
2. Preserve ALL factual information — do not invent experiences, companies, or dates
3. Strengthen and quantify bullet points (use [X] or [N] as a placeholder where a number is needed)
4. Add role-appropriate keywords naturally within existing experience descriptions
5. Rewrite the summary to reflect the target role transition (2-3 sentences, no first-person pronouns)

Output the complete rewritten resume using EXACTLY this structure (no other text before or after):

## CONTACT
[Full Name]
[email] | [phone] | [location]
[linkedin if present]

## SUMMARY
[2-3 sentences targeted to role, no "I" pronoun]

## EXPERIENCE

### [Job Title] | [Company] | [Start Date] – [End Date]
- [rewritten bullet with strong verb + quantification]
- [rewritten bullet]
[repeat for all roles]

## SKILLS
[Comma-separated skills, adding role-relevant terms where genuine]

## EDUCATION
[Degree] — [Institution], [Year]

## CERTIFICATIONS
- [cert]
[omit section if none]
"""


def _build_cover_letter_prompt(
    resume_data: ResumeData,
    job_title: str,
    company: str,
    job_description: str,
    target_role: str,
) -> str:
    c = resume_data.contact
    recent = resume_data.experience[0] if resume_data.experience else None
    recent_title = f"{recent.title} at {recent.company}" if recent else "IT Professional"

    top_bullets = []
    for exp in resume_data.experience[:2]:
        top_bullets.extend(exp.bullets[:2])
    bullets_text = "\n".join(f"• {b}" for b in top_bullets) or "• Extensive IT Operations experience"

    role_label = ROLE_LABELS.get(target_role, target_role.replace("_", " ").title())
    today = date.today().strftime("%B %d, %Y")

    return f"""Write a cover letter for this job application:

POSITION: {job_title} at {company}

JOB DESCRIPTION (excerpt):
{job_description[:1500]}

CANDIDATE BACKGROUND:
Most Recent Role: {recent_title}
Target Role: {role_label}
Core Skills: {', '.join(resume_data.skills[:12]) or 'IT Operations, Systems Management'}
Key Achievements:
{bullets_text}
Certifications: {', '.join(resume_data.certifications) if resume_data.certifications else 'None listed'}

REQUIREMENTS:
1. Open with a specific, compelling hook — NOT "I am writing to express my interest"
2. Paragraph 1: hook + why this specific company/role resonates
3. Paragraph 2: how IT Operations background directly transfers to this role (2-3 specific examples)
4. Paragraph 3: why this transition makes sense now and what unique value the IT background brings
5. Closing paragraph: enthusiasm + clear call to action
6. Maximum 4 paragraphs, under 400 words total
7. Professional but warm tone — no stiff corporate jargon
8. Do NOT use markdown — plain text only

Today's date: {today}
Sign off with: {c.name or 'The Candidate'}
"""


async def optimize_resume_streaming(
    client: AsyncAnthropic,
    resume_data: ResumeData,
    target_role: str,
    job_description: Optional[str],
) -> AsyncIterator[str]:
    user_prompt = _build_resume_user_prompt(resume_data, job_description)
    async with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=_build_system_blocks(target_role),
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def optimize_resume_full(
    client: AsyncAnthropic,
    resume_data: ResumeData,
    target_role: str,
    job_description: Optional[str],
) -> str:
    user_prompt = _build_resume_user_prompt(resume_data, job_description)
    response = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=_build_system_blocks(target_role),
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text


async def generate_cover_letter_streaming(
    client: AsyncAnthropic,
    request: CoverLetterRequest,
) -> AsyncIterator[str]:
    system_blocks: list[dict] = [
        {
            "type": "text",
            "text": "You are an expert cover letter writer specializing in IT-to-tech career transitions. "
            "Write compelling, concise cover letters that are warm and professional. "
            "Never use generic openers. Connect the candidate's IT Operations background to the role's needs. "
            "Output plain text only — no markdown, no bullet points in the cover letter itself.",
        },
    ]
    if request.target_role in ROLE_PROFILES:
        system_blocks.append(
            {
                "type": "text",
                "text": ROLE_PROFILES[request.target_role],
                "cache_control": {"type": "ephemeral"},
            }
        )

    user_prompt = _build_cover_letter_prompt(
        request.resume_data,
        request.job_title,
        request.company,
        request.job_description,
        request.target_role,
    )

    async with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system_blocks,
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        async for text in stream.text_stream:
            yield text

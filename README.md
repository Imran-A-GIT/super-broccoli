# CareerLift — AI Resume & Job Application Tool

A web-based AI-powered resume optimizer and job search platform built for IT Operations professionals transitioning into Tech Sales, Product Owner, Cybersecurity, or Data Analyst roles.

## Features

- **Resume Parsing** — Upload PDF, DOCX, or paste text; auto-extracts contact info, experience, skills, and education
- **ATS Score & Gap Analysis** — Score your resume against any job description, identify missing keywords, get actionable recommendations
- **AI Resume Optimization** — Claude AI rewrites your resume for your target role, adding role-specific keywords while preserving all facts
- **Cover Letter Generator** — Streaming AI-generated cover letters tailored to specific job postings
- **DOCX Export** — Download ATS-optimized DOCX files ready to submit
- **Job Search** — Search Adzuna and The Muse for high-paying jobs, sorted by salary, with resume match scores
- **Application Tracker** — Track every application with status updates, follow-up reminders, and dashboard stats

## Quick Start

### 1. Install

```bash
cd super-broccoli
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` with your keys:

```
ANTHROPIC_API_KEY=sk-ant-...      # Required for AI features — get at console.anthropic.com
ADZUNA_APP_ID=your_id              # Optional — get free at developer.adzuna.com
ADZUNA_APP_KEY=your_key            # Optional — needed for job search salary data
DATABASE_URL=sqlite+aiosqlite:///./applications.db
```

### 3. Run

```bash
uvicorn backend.main:app --reload --port 8000
```

Open `http://localhost:8000` in your browser.

## API Keys

| Key | Required | Where to get |
|-----|----------|--------------|
| `ANTHROPIC_API_KEY` | Yes (for AI features) | [console.anthropic.com](https://console.anthropic.com) |
| `ADZUNA_APP_ID` + `ADZUNA_APP_KEY` | Optional (job search) | [developer.adzuna.com](https://developer.adzuna.com) — free tier |

The tool works without Adzuna keys — The Muse API requires no key.

## Target Roles

The optimizer has specialized profiles for transitioning from IT Operations to:

- **Tech Sales / Solutions Engineer** — Client relationships, technical advisory, solution selling, SLA management as customer success
- **Product Owner / PM** — IT work reframed as backlog management, sprint planning, stakeholder coordination, OKR/KPI
- **Cybersecurity Analyst** — Security-adjacent IT work highlighted (firewalls, IAM, incident handling), SIEM/NIST/SOC keywords added
- **Data Analyst / BI** — Metrics, reporting, monitoring reframed as data analysis; SQL, dashboards, forecasting keywords

## Architecture

```
backend/
├── main.py                  # FastAPI app
├── config.py                # Settings (pydantic-settings + .env)
├── database.py              # SQLAlchemy async + SQLite
├── models.py                # Pydantic schemas
├── services/
│   ├── resume_parser.py     # PDF/DOCX/text parsing
│   ├── ats_scorer.py        # Keyword scoring + Claude semantic matching
│   ├── resume_optimizer.py  # Claude streaming rewrite + cover letter
│   ├── job_searcher.py      # Adzuna + The Muse API integration
│   └── exporter.py          # ATS-safe DOCX export
└── routers/
    ├── resume.py            # /api/resume/*
    ├── jobs.py              # /api/jobs/*
    └── applications.py      # /api/applications/*
frontend/
└── index.html              # 4-tab SPA (Tailwind CSS + vanilla JS)
```

## API Docs

Interactive docs at `http://localhost:8000/docs` after starting the server.

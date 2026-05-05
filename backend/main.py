from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .database import init_db
from .routers import applications, jobs, resume


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="CareerLift — Resume & Job Application Tool",
    description="AI-powered resume optimization and job search for IT Operations professionals",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(resume.router)
app.include_router(jobs.router)
app.include_router(applications.router)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
FRONTEND_INDEX = os.path.join(FRONTEND_DIR, "index.html")


@app.get("/", response_class=FileResponse, include_in_schema=False)
async def serve_spa():
    return FileResponse(FRONTEND_INDEX)


@app.get("/{full_path:path}", response_class=FileResponse, include_in_schema=False)
async def serve_spa_routes(full_path: str):
    return FileResponse(FRONTEND_INDEX)

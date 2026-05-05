from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import Application, get_session
from ..models import ApplicationCreate, ApplicationOut, ApplicationUpdate, DashboardStats

router = APIRouter(prefix="/api/applications", tags=["applications"])


@router.get("/dashboard", response_model=DashboardStats)
async def dashboard(session: AsyncSession = Depends(get_session)):
    # Count by status
    stmt = select(Application.status, func.count(Application.id)).group_by(Application.status)
    result = await session.execute(stmt)
    by_status: dict[str, int] = {row[0]: row[1] for row in result.all()}

    total = sum(by_status.values())
    positive = sum(
        by_status.get(s, 0) for s in ("Phone Screen", "Interview", "Offer")
    )
    response_rate = round(positive / total, 3) if total > 0 else 0.0

    # Follow-ups due within 3 days
    cutoff = datetime.utcnow() + timedelta(days=3)
    fu_stmt = (
        select(Application)
        .where(Application.follow_up_date <= cutoff)
        .where(Application.follow_up_date >= datetime.utcnow())
        .order_by(Application.follow_up_date)
    )
    fu_result = await session.execute(fu_stmt)
    follow_ups = fu_result.scalars().all()

    return DashboardStats(
        total=total,
        by_status=by_status,
        response_rate=response_rate,
        interviews_scheduled=by_status.get("Interview", 0),
        offers=by_status.get("Offer", 0),
        follow_ups_due=[ApplicationOut.model_validate(a) for a in follow_ups],
    )


@router.get("/", response_model=list[ApplicationOut])
async def list_applications(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
):
    stmt = select(Application).order_by(Application.created_at.desc())
    if status:
        stmt = stmt.where(Application.status == status)
    stmt = stmt.limit(limit).offset(offset)
    result = await session.execute(stmt)
    return [ApplicationOut.model_validate(a) for a in result.scalars().all()]


@router.post("/", response_model=ApplicationOut, status_code=201)
async def create_application(
    body: ApplicationCreate,
    session: AsyncSession = Depends(get_session),
):
    app = Application(**body.model_dump(exclude_none=True))
    session.add(app)
    await session.commit()
    await session.refresh(app)
    return ApplicationOut.model_validate(app)


@router.get("/{app_id}", response_model=ApplicationOut)
async def get_application(app_id: int, session: AsyncSession = Depends(get_session)):
    app = await session.get(Application, app_id)
    if not app:
        raise HTTPException(status_code=404, detail="Application not found")
    return ApplicationOut.model_validate(app)


@router.patch("/{app_id}", response_model=ApplicationOut)
async def update_application(
    app_id: int,
    body: ApplicationUpdate,
    session: AsyncSession = Depends(get_session),
):
    app = await session.get(Application, app_id)
    if not app:
        raise HTTPException(status_code=404, detail="Application not found")
    for field, value in body.model_dump(exclude_none=True).items():
        setattr(app, field, value)
    await session.commit()
    await session.refresh(app)
    return ApplicationOut.model_validate(app)


@router.delete("/{app_id}")
async def delete_application(app_id: int, session: AsyncSession = Depends(get_session)):
    app = await session.get(Application, app_id)
    if not app:
        raise HTTPException(status_code=404, detail="Application not found")
    await session.delete(app)
    await session.commit()
    return {"ok": True}

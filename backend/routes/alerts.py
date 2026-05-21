import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import Alert
from schemas import AlertCreate, AlertResponse

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("/", response_model=list[AlertResponse])
async def get_alerts(
    resolved: bool | None = None, db: AsyncSession = Depends(get_db)
):
    query = select(Alert).order_by(Alert.created_at.desc())
    if resolved is not None:
        query = query.where(Alert.is_resolved == resolved)
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/active", response_model=list[AlertResponse])
async def get_active_alerts(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Alert).where(Alert.is_resolved == False).order_by(Alert.created_at.desc())  # noqa: E712
    )
    return result.scalars().all()


@router.post("/", response_model=AlertResponse)
async def create_alert(alert_data: AlertCreate, db: AsyncSession = Depends(get_db)):
    alert = Alert(**alert_data.model_dump())
    db.add(alert)
    await db.commit()
    await db.refresh(alert)
    return alert


@router.put("/{alert_id}/resolve")
async def resolve_alert(alert_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert.is_resolved = True
    alert.resolved_at = datetime.datetime.utcnow()
    await db.commit()
    return {"message": "Alert resolved", "alert_id": alert_id}

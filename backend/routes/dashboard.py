import datetime

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import Alert, Boat, BoatStatus, Trip
from schemas import DashboardStats

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/stats", response_model=DashboardStats)
async def get_dashboard_stats(db: AsyncSession = Depends(get_db)):
    total = await db.execute(select(func.count(Boat.id)))
    total_boats = total.scalar() or 0

    active = await db.execute(
        select(func.count(Boat.id)).where(Boat.status == BoatStatus.ACTIVE)
    )
    active_boats = active.scalar() or 0

    docked = await db.execute(
        select(func.count(Boat.id)).where(Boat.status == BoatStatus.DOCKED)
    )
    docked_boats = docked.scalar() or 0

    sos = await db.execute(
        select(func.count(Boat.id)).where(Boat.status == BoatStatus.SOS)
    )
    sos_boats = sos.scalar() or 0

    alerts = await db.execute(
        select(func.count(Alert.id)).where(Alert.is_resolved == False)  # noqa: E712
    )
    active_alerts = alerts.scalar() or 0

    today = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    trips_today = await db.execute(
        select(func.count(Trip.id)).where(Trip.start_time >= today)
    )
    total_trips_today = trips_today.scalar() or 0

    distance_today = await db.execute(
        select(func.sum(Trip.distance_km)).where(Trip.start_time >= today)
    )
    total_distance_today = distance_today.scalar() or 0.0

    return DashboardStats(
        total_boats=total_boats,
        active_boats=active_boats,
        docked_boats=docked_boats,
        sos_boats=sos_boats,
        active_alerts=active_alerts,
        total_trips_today=total_trips_today,
        total_distance_today=total_distance_today,
    )

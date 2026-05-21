from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import Trip
from schemas import TripResponse

router = APIRouter(prefix="/api/trips", tags=["trips"])


@router.get("/", response_model=list[TripResponse])
async def get_trips(boat_id: int | None = None, db: AsyncSession = Depends(get_db)):
    query = select(Trip).order_by(Trip.start_time.desc())
    if boat_id:
        query = query.where(Trip.boat_id == boat_id)
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/active", response_model=list[TripResponse])
async def get_active_trips(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Trip).where(Trip.is_active == True).order_by(Trip.start_time.desc())  # noqa: E712
    )
    return result.scalars().all()

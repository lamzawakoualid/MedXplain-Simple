import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import Boat, BoatStatus
from schemas import BoatCreate, BoatResponse, BoatUpdate

router = APIRouter(prefix="/api/boats", tags=["boats"])


@router.get("/", response_model=list[BoatResponse])
async def get_boats(status: BoatStatus | None = None, db: AsyncSession = Depends(get_db)):
    query = select(Boat)
    if status:
        query = query.where(Boat.status == status)
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/{boat_id}", response_model=BoatResponse)
async def get_boat(boat_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Boat).where(Boat.id == boat_id))
    boat = result.scalar_one_or_none()
    if not boat:
        raise HTTPException(status_code=404, detail="Boat not found")
    return boat


@router.post("/", response_model=BoatResponse)
async def create_boat(boat_data: BoatCreate, db: AsyncSession = Depends(get_db)):
    boat = Boat(**boat_data.model_dump())
    db.add(boat)
    await db.commit()
    await db.refresh(boat)
    return boat


@router.put("/{boat_id}/position", response_model=BoatResponse)
async def update_boat_position(
    boat_id: int, update: BoatUpdate, db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(Boat).where(Boat.id == boat_id))
    boat = result.scalar_one_or_none()
    if not boat:
        raise HTTPException(status_code=404, detail="Boat not found")

    boat.latitude = update.latitude
    boat.longitude = update.longitude
    boat.speed = update.speed
    boat.heading = update.heading
    boat.battery_level = update.battery_level
    boat.status = update.status
    boat.last_seen = datetime.datetime.utcnow()

    await db.commit()
    await db.refresh(boat)
    return boat


@router.post("/{boat_id}/sos")
async def trigger_sos(boat_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Boat).where(Boat.id == boat_id))
    boat = result.scalar_one_or_none()
    if not boat:
        raise HTTPException(status_code=404, detail="Boat not found")

    boat.status = BoatStatus.SOS
    boat.last_seen = datetime.datetime.utcnow()
    await db.commit()
    return {"message": f"SOS triggered for boat {boat.name}", "boat_id": boat_id}

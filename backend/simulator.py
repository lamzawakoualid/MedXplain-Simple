"""
IoT GPS Simulator - Simulates boat positions along Morocco's Atlantic coast.
Generates realistic GPS data for testing the SmartSea tracking system.
"""

import asyncio
import datetime
import math
import random

from sqlalchemy import select

from database import async_session, init_db
from models import Alert, AlertType, Boat, BoatStatus, PositionLog, Trip

# Moroccan ports with coordinates
MOROCCAN_PORTS = [
    {"name": "Boujdour", "lat": 26.1285, "lng": -14.4967},
    {"name": "Dakhla", "lat": 23.6848, "lng": -15.9570},
    {"name": "Laayoune", "lat": 27.1536, "lng": -13.1990},
    {"name": "Tan-Tan", "lat": 28.4380, "lng": -11.1033},
    {"name": "Agadir", "lat": 30.4278, "lng": -9.5981},
    {"name": "Essaouira", "lat": 31.5085, "lng": -9.7595},
    {"name": "Safi", "lat": 32.2994, "lng": -9.2372},
    {"name": "El Jadida", "lat": 33.2316, "lng": -8.5007},
    {"name": "Casablanca", "lat": 33.5731, "lng": -7.5898},
    {"name": "Rabat", "lat": 34.0209, "lng": -6.8416},
]

BOAT_NAMES = [
    "بركة البحر", "نجمة الأطلسي", "أمل الصياد", "رياح الجنوب",
    "فجر البحر", "سفينة النور", "قارب السعادة", "موج الأمان",
    "نسيم البحر", "زورق الحرية", "صقر المحيط", "لؤلؤة الساحل",
]


async def seed_boats():
    """Create sample boats in the database."""
    async with async_session() as db:
        result = await db.execute(select(Boat))
        existing = result.scalars().all()
        if existing:
            return

        for i, name in enumerate(BOAT_NAMES):
            port = MOROCCAN_PORTS[i % len(MOROCCAN_PORTS)]
            boat = Boat(
                name=name,
                registration_number=f"MA-{port['name'][:3].upper()}-{1000 + i}",
                owner_name=f"صياد {i + 1}",
                phone_number=f"+2126{random.randint(10000000, 99999999)}",
                port=port["name"],
                latitude=port["lat"],
                longitude=port["lng"],
                speed=0.0,
                heading=random.uniform(0, 360),
                battery_level=random.randint(70, 100),
                status=BoatStatus.DOCKED,
            )
            db.add(boat)

        await db.commit()
        print(f"✓ Seeded {len(BOAT_NAMES)} boats")


async def simulate_movement(boat_id: int, base_lat: float, base_lng: float):
    """Simulate realistic boat movement from a port."""
    angle = random.uniform(0, 2 * math.pi)
    speed = random.uniform(3, 12)  # knots
    max_distance = 0.5  # degrees from port

    while True:
        async with async_session() as db:
            result = await db.execute(select(Boat).where(Boat.id == boat_id))
            boat = result.scalar_one_or_none()
            if not boat:
                return

            if boat.status == BoatStatus.DOCKED:
                if random.random() < 0.3:
                    boat.status = BoatStatus.ACTIVE
                    trip = Trip(
                        boat_id=boat_id,
                        start_time=datetime.datetime.utcnow(),
                        start_latitude=boat.latitude,
                        start_longitude=boat.longitude,
                    )
                    db.add(trip)
                await db.commit()
                await asyncio.sleep(random.uniform(5, 15))
                continue

            # Random course changes
            angle += random.uniform(-0.3, 0.3)
            speed = max(1, min(15, speed + random.uniform(-1, 1)))

            # Calculate new position
            dt = 10 / 3600  # 10 seconds in hours
            dlat = speed * math.cos(angle) * dt / 60
            dlng = speed * math.sin(angle) * dt / 60

            new_lat = boat.latitude + dlat
            new_lng = boat.longitude + dlng

            # Keep within range of port
            dist_from_port = math.sqrt((new_lat - base_lat) ** 2 + (new_lng - base_lng) ** 2)
            if dist_from_port > max_distance:
                angle = math.atan2(base_lat - new_lat, base_lng - new_lng)
                new_lat = boat.latitude + speed * math.cos(angle) * dt / 60
                new_lng = boat.longitude + speed * math.sin(angle) * dt / 60

            boat.latitude = new_lat
            boat.longitude = new_lng
            boat.speed = round(speed, 1)
            boat.heading = round(math.degrees(angle) % 360, 1)
            boat.battery_level = max(10, boat.battery_level - random.randint(0, 1))
            boat.last_seen = datetime.datetime.utcnow()

            # Log position
            log = PositionLog(
                boat_id=boat_id,
                latitude=new_lat,
                longitude=new_lng,
                speed=speed,
                heading=boat.heading,
            )
            db.add(log)

            # Random events
            if random.random() < 0.005:
                boat.status = BoatStatus.SOS
                alert = Alert(
                    boat_id=boat_id,
                    alert_type=AlertType.SOS,
                    message=f"إشارة استغاثة من القارب {boat.name}",
                    latitude=new_lat,
                    longitude=new_lng,
                )
                db.add(alert)

            if boat.battery_level < 20 and random.random() < 0.1:
                alert = Alert(
                    boat_id=boat_id,
                    alert_type=AlertType.LOW_BATTERY,
                    message=f"بطارية منخفضة: {boat.battery_level}%",
                    latitude=new_lat,
                    longitude=new_lng,
                )
                db.add(alert)

            if speed > 12 and random.random() < 0.1:
                alert = Alert(
                    boat_id=boat_id,
                    alert_type=AlertType.SPEED,
                    message=f"سرعة مرتفعة: {speed:.1f} عقدة",
                    latitude=new_lat,
                    longitude=new_lng,
                )
                db.add(alert)

            # Random return to port
            if random.random() < 0.02:
                boat.status = BoatStatus.DOCKED
                boat.latitude = base_lat
                boat.longitude = base_lng
                boat.speed = 0

                # End active trip
                trip_result = await db.execute(
                    select(Trip)
                    .where(Trip.boat_id == boat_id, Trip.is_active == True)  # noqa: E712
                    .order_by(Trip.start_time.desc())
                )
                active_trip = trip_result.scalar_one_or_none()
                if active_trip:
                    active_trip.is_active = False
                    active_trip.end_time = datetime.datetime.utcnow()
                    active_trip.end_latitude = base_lat
                    active_trip.end_longitude = base_lng
                    active_trip.distance_km = round(random.uniform(5, 50), 1)
                    active_trip.max_speed = round(speed + random.uniform(2, 5), 1)
                    active_trip.avg_speed = round(speed * 0.7, 1)

            await db.commit()

        await asyncio.sleep(random.uniform(3, 8))


async def run_simulator():
    """Run the full simulation."""
    await init_db()
    await seed_boats()

    async with async_session() as db:
        result = await db.execute(select(Boat))
        boats = result.scalars().all()

    tasks = []
    for boat in boats:
        port = next(
            (p for p in MOROCCAN_PORTS if p["name"] == boat.port),
            MOROCCAN_PORTS[0],
        )
        tasks.append(simulate_movement(boat.id, port["lat"], port["lng"]))

    print(f"✓ Simulator running for {len(boats)} boats")
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(run_simulator())

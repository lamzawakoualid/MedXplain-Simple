import datetime

from pydantic import BaseModel

from models import AlertType, BoatStatus


class BoatCreate(BaseModel):
    name: str
    registration_number: str
    owner_name: str
    phone_number: str | None = None
    port: str | None = None


class BoatUpdate(BaseModel):
    latitude: float
    longitude: float
    speed: float = 0.0
    heading: float = 0.0
    battery_level: int = 100
    status: BoatStatus = BoatStatus.ACTIVE


class BoatResponse(BaseModel):
    id: int
    name: str
    registration_number: str
    owner_name: str
    phone_number: str | None
    port: str | None
    latitude: float
    longitude: float
    speed: float
    heading: float
    battery_level: int
    status: BoatStatus
    last_seen: datetime.datetime
    created_at: datetime.datetime

    class Config:
        from_attributes = True


class AlertCreate(BaseModel):
    boat_id: int
    alert_type: AlertType
    message: str | None = None
    latitude: float | None = None
    longitude: float | None = None


class AlertResponse(BaseModel):
    id: int
    boat_id: int
    alert_type: AlertType
    message: str | None
    latitude: float | None
    longitude: float | None
    is_resolved: bool
    created_at: datetime.datetime
    resolved_at: datetime.datetime | None

    class Config:
        from_attributes = True


class TripResponse(BaseModel):
    id: int
    boat_id: int
    start_time: datetime.datetime
    end_time: datetime.datetime | None
    start_latitude: float | None
    start_longitude: float | None
    end_latitude: float | None
    end_longitude: float | None
    distance_km: float
    max_speed: float
    avg_speed: float
    is_active: bool

    class Config:
        from_attributes = True


class DashboardStats(BaseModel):
    total_boats: int
    active_boats: int
    docked_boats: int
    sos_boats: int
    active_alerts: int
    total_trips_today: int
    total_distance_today: float


class PositionUpdate(BaseModel):
    boat_id: int
    latitude: float
    longitude: float
    speed: float = 0.0
    heading: float = 0.0
    battery_level: int = 100

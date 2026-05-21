import datetime
import enum

from sqlalchemy import Boolean, Column, DateTime, Enum, Float, Integer, String, Text

from database import Base


class BoatStatus(str, enum.Enum):
    ACTIVE = "active"
    DOCKED = "docked"
    SOS = "sos"
    OFFLINE = "offline"


class AlertType(str, enum.Enum):
    SOS = "sos"
    WEATHER = "weather"
    SPEED = "speed"
    BOUNDARY = "boundary"
    LOW_BATTERY = "low_battery"


class Boat(Base):
    __tablename__ = "boats"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    registration_number = Column(String(50), unique=True, nullable=False)
    owner_name = Column(String(100), nullable=False)
    phone_number = Column(String(20))
    port = Column(String(100))
    latitude = Column(Float, default=0.0)
    longitude = Column(Float, default=0.0)
    speed = Column(Float, default=0.0)
    heading = Column(Float, default=0.0)
    battery_level = Column(Integer, default=100)
    status = Column(Enum(BoatStatus), default=BoatStatus.DOCKED)
    last_seen = Column(DateTime, default=datetime.datetime.utcnow)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    boat_id = Column(Integer, nullable=False)
    alert_type = Column(Enum(AlertType), nullable=False)
    message = Column(Text)
    latitude = Column(Float)
    longitude = Column(Float)
    is_resolved = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)


class Trip(Base):
    __tablename__ = "trips"

    id = Column(Integer, primary_key=True, index=True)
    boat_id = Column(Integer, nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    start_latitude = Column(Float)
    start_longitude = Column(Float)
    end_latitude = Column(Float, nullable=True)
    end_longitude = Column(Float, nullable=True)
    distance_km = Column(Float, default=0.0)
    max_speed = Column(Float, default=0.0)
    avg_speed = Column(Float, default=0.0)
    is_active = Column(Boolean, default=True)


class PositionLog(Base):
    __tablename__ = "position_logs"

    id = Column(Integer, primary_key=True, index=True)
    boat_id = Column(Integer, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    speed = Column(Float, default=0.0)
    heading = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

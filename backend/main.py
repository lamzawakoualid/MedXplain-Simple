import asyncio
import contextlib
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from database import init_db
from routes import alerts, boats, dashboard, trips, websocket
from simulator import run_simulator

simulator_task = None


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    global simulator_task
    simulator_task = asyncio.create_task(run_simulator())
    yield
    if simulator_task:
        simulator_task.cancel()


app = FastAPI(
    title="SmartSea Morocco API",
    description="نظام ذكي لتتبع قوارب الصيد التقليدي بالمغرب",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(boats.router)
app.include_router(alerts.router)
app.include_router(trips.router)
app.include_router(dashboard.router)
app.include_router(websocket.router)


@app.get("/health")
async def health():
    return {"status": "healthy"}


# Serve frontend static files if available
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
else:

    @app.get("/")
    async def root():
        return {
            "name": "SmartSea Morocco API",
            "version": "0.1.0",
            "description": "Smart tracking system for traditional fishing boats in Morocco",
        }

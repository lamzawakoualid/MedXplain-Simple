# 🚢 SmartSea Morocco

**نظام ذكي لتتبع قوارب الصيد التقليدي بالمغرب**

Smart tracking system for traditional fishing boats in Morocco using GPS, IoT, and AI.

![SmartSea](https://img.shields.io/badge/SmartSea-Morocco-0ea5e9?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black)

## 🌟 Features

- **Real-time Tracking** - Live boat positions on an interactive map
- **SOS Emergency System** - One-click emergency alerts
- **Weather Alerts** - Storm and dangerous condition warnings
- **Trip Analytics** - Distance, speed, and route analysis
- **Battery Monitoring** - Solar-powered device status
- **Port Dashboard** - Overview for harbor authorities

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   IoT Device    │────▶│   FastAPI Backend │────▶│  React Frontend │
│  (ESP32 + GPS)  │     │   + WebSocket     │     │  + Leaflet Map  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                        ┌──────┴──────┐
                        │  SQLite DB  │
                        │ (PostgreSQL │
                        │  in prod)   │
                        └─────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- pip or uv

### Backend Setup

```bash
cd backend
pip install -e .
uvicorn main:app --reload --port 8000
```

The backend includes a built-in IoT simulator that generates realistic GPS data for 12 fishing boats along Morocco's Atlantic coast.

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000 to view the dashboard.

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/boats/` | List all boats |
| GET | `/api/boats/{id}` | Get boat details |
| POST | `/api/boats/` | Register new boat |
| PUT | `/api/boats/{id}/position` | Update position |
| POST | `/api/boats/{id}/sos` | Trigger SOS |
| GET | `/api/alerts/active` | Active alerts |
| POST | `/api/alerts/` | Create alert |
| PUT | `/api/alerts/{id}/resolve` | Resolve alert |
| GET | `/api/trips/` | List trips |
| GET | `/api/dashboard/stats` | Dashboard stats |
| WS | `/ws/tracking` | Real-time updates |

## 🗺️ Simulated Ports

The prototype simulates boats across major Moroccan fishing ports:

- Boujdour, Dakhla, Laayoune
- Tan-Tan, Agadir, Essaouira
- Safi, El Jadida, Casablanca, Rabat

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python + FastAPI |
| Frontend | React + Vite |
| Database | SQLite (dev) / PostgreSQL (prod) |
| Maps | Leaflet + CARTO Dark |
| Real-time | WebSocket |
| IoT Sim | Python asyncio |

## 📱 IoT Hardware (Production)

| Component | Function |
|-----------|----------|
| ESP32 | Processing & connectivity |
| GPS NEO-6M | Position tracking |
| SIM800L | Data transmission (GSM) |
| Solar Panel | Power supply |
| SOS Button | Emergency trigger |

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

**SmartSea Morocco** - حماية البحارة، رقمنة الصيد التقليدي 🇲🇦

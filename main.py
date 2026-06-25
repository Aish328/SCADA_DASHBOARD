from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from database import engine
from sqlalchemy import create_engine, text
import pandas as pd

from routes import kpi, filters, spikes, data, ml


app = FastAPI(title="SCADA Intelligence Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ml.router, prefix="/ml")
app.include_router(kpi.router, prefix="/kpi")
app.include_router(filters.router, prefix="/filters")
app.include_router(spikes.router, prefix="/spikes")
app.include_router(data.router, prefix="/data")

# =========================
# Static Files
# =========================
static_dir = Path(__file__).parent / "static"

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    print("[INFO] Static files mounted successfully")
else:
    print("[WARN] Static directory not found")

# =========================
# Startup Event (IMPORTANT)
# =========================
@app.on_event("startup")
def startup_event():
    print("\n==============================")
    print("[STARTUP] SCADA Backend Booting...")
    print("[STARTUP] Connecting to PostgreSQL...")

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("[SUCCESS] PostgreSQL connection established ✔")
    except Exception as e:
        print("[ERROR] PostgreSQL connection failed ❌")
        print(e)

    print("==============================\n")

# =========================
# Routes
# =========================
@app.get("/")
def serve_index():
    print("[REQUEST] GET /")
    return FileResponse(static_dir / "index.html")

@app.get("/health")
def health():
    print("[REQUEST] GET /health")
    return {"status": "ok"}

@app.get("/sensor-data")
def get_sensor_data():
    print("[REQUEST] GET /sensor-data")
    print("[DB] Fetching sensor_data from PostgreSQL...")

    try:
        query = engine.connect().execute(text("SELECT * FROM scada_db"))
        df = pd.read_sql(query, engine)

        print(f"[DB SUCCESS] Fetched {len(df)} rows ✔")
        return df.to_dict(orient="records")

    except Exception as e:
        print("[DB ERROR] Failed to fetch sensor_data ❌")
        print(e)
        return {"error": str(e)}

@app.get("/ml")
def ml_page():
    print("[REQUEST] GET /ml")
    return FileResponse(static_dir / "ml.html")
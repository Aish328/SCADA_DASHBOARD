from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from routes import kpi, filters, spikes, data, ml

from sqlalchemy import create_engine
import pandas as pd

# PostgreSQL connection
DATABASE_URL = "postgresql://postgres:admin123@localhost:5432/scada_db"

engine = create_engine(DATABASE_URL)

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

# Serve frontend static files
static_dir = Path(__file__).parent / "static"

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
def serve_index():
    index = Path(__file__).parent / "static" / "index.html"
    return FileResponse(str(index))

@app.get("/health")
def health():
    return {"status": "ok"}

# PostgreSQL API
@app.get("/sensor-data")
def get_sensor_data():

    query = "SELECT * FROM sensor_data LIMIT 100"

    df = pd.read_sql(query, engine)

    return df.to_dict(orient="records")
@app.get("/ml")
def ml_page():
    return FileResponse("static/ml.html") 
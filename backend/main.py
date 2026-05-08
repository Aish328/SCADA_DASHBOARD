from fastapi import FastAPI
from backend.routes import kpi, filters, spikes

# from SCADA_DASHBOARD.SCADA_DASHBOARD.backend.routes import kpi , filters, spikes
app = FastAPI()

app.include_router(kpi.router, prefix = "/kpi")
app.include_router(filters.router , prefix="/filters")
app.include_router(spikes.router , prefix = "/spikes")

@app.get("/")
def home():
    return {"message" : "SCADA Dashboard API running"}
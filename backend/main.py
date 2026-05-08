from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import kpi, filters, spikes, data

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(kpi.router, prefix="/kpi")
app.include_router(filters.router, prefix="/filters")
app.include_router(spikes.router, prefix="/spikes")
app.include_router(data.router, prefix="/data")

@app.get("/")
def home():
    return {"message": "SCADA Dashboard API running"}
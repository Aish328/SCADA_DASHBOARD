from fastapi import APIRouter
import pandas as pd

router = APIRouter()

@router.get("/")
def get_kpis():
    df = pd.read_csv(r"C:\Users\sharika\Downloads\scada_large_dataset.csv")


    return {
        "avg_voltage": float(df['voltage'].mean()),
        "max_current": float(df['current'].max()),
        "total_spikes": int((df['power'] > 100).mean()),
        "total_records" : len(df)
    }
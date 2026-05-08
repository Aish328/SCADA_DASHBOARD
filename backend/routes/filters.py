from fastapi import FastAPI
import pandas as pd
from fastapi import APIRouter

router = APIRouter()
@router.get("/")
def get_filters():
    df = pd.read_csv(r"C:\Users\sharika\Downloads\scada_large_dataset.csv")
    return{
        "cities" : df['city'].unique().tolist(),
        "divisions" : df['division'].unique().tolist(),
        "substations" : df['substation'].unique().tolist(),
        "feeders" : df['feeder'].unique().tolist()

    }
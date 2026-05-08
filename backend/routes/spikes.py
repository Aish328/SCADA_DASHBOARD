from fastapi import FastAPI
import pandas as pd
from fastapi import APIRouter

router = APIRouter()
def get_spikes():
    df = pd.read_csv(r"C:\Users\sharika\Downloads\scada_large_dataset.csv")

    spikes = df[df["power"] > 250]

    return spikes.to_dict(orient = "records")


                     
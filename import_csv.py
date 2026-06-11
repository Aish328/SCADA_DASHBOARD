import pandas as pd
from sqlalchemy import create_engine

# PostgreSQL connection
DATABASE_URL = "postgresql://postgres:admin123@localhost:5432/scada_db"

engine = create_engine(DATABASE_URL)

# Read CSV
df = pd.read_csv(r"C:\Users\Aishanya\Desktop\scada_dashboard\SCADA_DASHBOARD\data\data.csv")

# Convert timestamp if present
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])

# Upload to PostgreSQL
df.to_sql(
    name="sensor_data",
    con=engine,
    if_exists="replace",   # use append later
    index=False
)

print("CSV imported successfully!")
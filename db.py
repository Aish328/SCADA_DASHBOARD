from sqlalchemy import create_engine
import pandas as pd

DATABASE_URL = "postgresql://postgres:admin123@localhost:5432/scada_db"

# SQLAlchemy engine
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

def get_raw_sensor_data(limit=100):
    """
    Fetch raw sensor data from PostgreSQL.
    """
    query = f"""
        SELECT *
        FROM sensor_data
        ORDER BY timestamp DESC
        LIMIT {limit}
    """
    df = pd.read_sql(query, engine)
    return df
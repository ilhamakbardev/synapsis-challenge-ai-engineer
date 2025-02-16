"""
RECOMMENDED to run it via docker instead of local.
"""

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import JSONResponse
from psycopg2 import connect
from psycopg2.extras import RealDictCursor
from typing import Optional
import uvicorn
from datetime import datetime, timedelta
import pandas as pd
from prophet import Prophet
import os

app = FastAPI()

DB_CONFIG = {
    'dbname': os.getenv("POSTGRES_DB", "vehicles-counter"),
    'user': os.getenv("POSTGRES_USER", "postgres"),
    'password': os.getenv("POSTGRES_PASSWORD", "admin"),
    'host': os.getenv("POSTGRES_HOST", "db"),
    'port': os.getenv("POSTGRES_PORT", "5432")
}

def get_db_connection():
    return connect(**DB_CONFIG)

@app.post("/api/areas/")
async def create_area(request: Request):
    data = await request.json()
    area_name = data.get("area_name")
    coordinates = data.get("coordinates")
    
    if not area_name or not coordinates:
        raise HTTPException(status_code=400, detail="Area name and coordinates are required")
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute('''
                    INSERT INTO area_configurations (area_name, coordinates)
                    VALUES (%s, %s)
                    RETURNING id;
                ''', (area_name, coordinates))
                result = cur.fetchone()
                
                if result is None:
                    raise HTTPException(status_code=500, detail="Failed to insert area")
                area_id = result[0]
                conn.commit()
            except Exception as e:
                print(f"Error during database operation: {e}")
                raise HTTPException(status_code=500, detail="Database error occurred")
        return {"id": area_id, "area_name": area_name, "coordinates": coordinates}
    finally:
        conn.close()

@app.post("/api/detections/")
async def create_detection_log(request: Request):
    try:
        log = await request.json()

        area_id = log.get("area_id")
        timestamp = log.get("timestamp")
        vehicle_id = log.get("vehicle_id")
        status = log.get("status")

        if not all([area_id, timestamp, vehicle_id, status]):
            raise HTTPException(status_code=400, detail="Missing required fields")

        try:
            timestamp = datetime.fromisoformat(timestamp)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid timestamp format")

        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM area_configurations WHERE id = %s", (area_id,))
            area_exists = cur.fetchone()

            if not area_exists:
                raise HTTPException(status_code=400, detail=f"Area with ID {area_id} does not exist.")

            cur.execute('''
                INSERT INTO vehicle_logs (timestamp, vehicle_id, status, area_id)
                VALUES (%s, %s, %s, %s)
                RETURNING id;
            ''', (timestamp, vehicle_id, status, area_id))

            conn.commit()
            return {"message": "Detection log inserted successfully", "log_id": cur.fetchone()[0]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats/")
def get_stats(start_time: Optional[str] = Query(None), end_time: Optional[str] = Query(None), page: int = 1, size: int = 10):
    with get_db_connection() as conn, conn.cursor() as cur:
        query = "SELECT timestamp, vehicle_id, status, area_id FROM vehicle_logs"
        params = []

        if start_time and end_time:
            query += " WHERE timestamp BETWEEN %s AND %s"
            params.extend([start_time, end_time])

        query += " ORDER BY timestamp DESC LIMIT %s OFFSET %s"
        params.extend([size, (page - 1) * size])

        cur.execute(query, tuple(params))
        logs = cur.fetchall()

    return [{"timestamp": str(row[0]), "vehicle_id": row[1], "status": row[2], "area_id": row[3]} for row in logs]

@app.get("/api/stats/live")
def get_live_stats():
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(DISTINCT vehicle_id) FROM vehicle_logs")
        total_vehicles = cur.fetchone()[0]

        one_second_ago = datetime.now() - timedelta(seconds=1)

        # Vehicles that entered in the last second
        cur.execute(
            "SELECT COUNT(DISTINCT vehicle_id) "
            "FROM vehicle_logs "
            "WHERE timestamp >= %s AND timestamp < %s AND status = 'entered'",
            (one_second_ago, datetime.now())
        )
        last_second_count_enter = cur.fetchone()[0]

        # Vehicles that exited in the last second
        cur.execute(
            "SELECT COUNT(DISTINCT vehicle_id) "
            "FROM vehicle_logs "
            "WHERE timestamp >= %s AND timestamp < %s AND status = 'exited'",
            (one_second_ago, datetime.now())
        )
        last_second_count_leave = cur.fetchone()[0]

    return {
        "total_vehicle_visited": total_vehicles,
        "last_visited_one_second": last_second_count_enter,
        "last_leave_one_second": last_second_count_leave
    }

@app.get("/api/forecast")
def get_forecast():
    with get_db_connection() as conn:
        df = pd.read_sql(
            "SELECT timestamp, COUNT(DISTINCT vehicle_id) as count FROM vehicle_logs WHERE status IN ('entered', 'exited') GROUP BY timestamp ORDER BY timestamp;", conn)
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])

    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=60, freq='min')
    forecast = model.predict(future)

    forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(
        orient='records')
    return {"forecast": forecast_data}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000,
                reload=True, log_level="info")

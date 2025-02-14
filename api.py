from fastapi import FastAPI, Query
from psycopg2 import connect
from typing import Optional
import uvicorn

app = FastAPI()

DB_CONFIG = {
    "dbname": "vehicles-counter",
    "user": "postgres",
    "password": "admin",
    "host": "127.0.0.1"
}

def get_db_connection():
    return connect(**DB_CONFIG)

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
        cur.execute("SELECT timestamp, vehicle_id, status, area_id FROM vehicle_logs ORDER BY timestamp DESC LIMIT 1")
        latest = cur.fetchone()
    
    if latest:
        return {"timestamp": str(latest[0]), "vehicle_id": latest[1], "status": latest[2], "area_id": latest[3]}
    return {"message": "No live data available."}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
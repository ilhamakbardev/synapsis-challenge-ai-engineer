from fastapi import FastAPI, Query
from psycopg2 import connect
from typing import Optional
import uvicorn
from datetime import datetime, timedelta

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
        # Total vehicle count by unique vehicle IDs
        cur.execute("SELECT COUNT(DISTINCT vehicle_id) FROM vehicle_logs")
        total_vehicles = cur.fetchone()[0]

        # Vehicles visited in the last 1 second
        one_second_ago = datetime.now() - timedelta(seconds=1)
        cur.execute(
            "SELECT COUNT(DISTINCT vehicle_id) "
            "FROM vehicle_logs "
            "WHERE timestamp >= %s AND status = 'entered'",
            (one_second_ago,)
        )
        last_second_count_enter = cur.fetchone()[0]

        # Vehicles leaved in the last 1 second
        one_second_ago = datetime.now() - timedelta(seconds=1)
        cur.execute(
            "SELECT COUNT(DISTINCT vehicle_id) "
            "FROM vehicle_logs "
            "WHERE timestamp >= %s AND status = 'exited'",
            (one_second_ago,)
        )
        last_second_count_leave = cur.fetchone()[0]

    return {
        "total_vehicle_visited": total_vehicles,
        "last_visited_one_second": last_second_count_enter,
        "last_leave_one_second": last_second_count_leave
    }


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000,
                reload=True, log_level="info")

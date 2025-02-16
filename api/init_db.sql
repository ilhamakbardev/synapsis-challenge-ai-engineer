CREATE TABLE IF NOT EXISTS area_configurations (
    id SERIAL PRIMARY KEY,
    area_name VARCHAR(255),
    coordinates TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vehicle_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP,
    vehicle_id VARCHAR(255),
    status VARCHAR(50),
    area_id INT,
    FOREIGN KEY (area_id) REFERENCES area_configurations(id)
);

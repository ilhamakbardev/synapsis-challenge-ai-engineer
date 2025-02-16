FROM python:3.11-slim

WORKDIR /app

COPY api.py requirements.txt ./

RUN apt-get update && apt-get install -y postgresql-client && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh", "-c", "until pg_isready -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER; do echo waiting for database; sleep 2; done && PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -f init_db.sql && uvicorn api:app --host 0.0.0.0 --port 8000"]

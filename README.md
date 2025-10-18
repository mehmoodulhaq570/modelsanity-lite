# ModelSanity Lite

A minimal template for a model sanity-checking web app.

## Structure

- `backend/` — Flask API server (Python)
- `frontend/` — React web client

## Quickstart

### Backend

```bash
cd modelsanity-lite/backend
pip install -r requirements.txt
python app.py
```

Or with Docker:

```bash
cd modelsanity-lite/backend
docker build -t modelsanity-backend .
docker run -p 5000:5000 modelsanity-backend
```

### Frontend

```bash
cd modelsanity-lite/frontend
npm install
npm start
```

## Deployment

- Use `Procfile` for Heroku or similar platforms.
- Dockerfile provided for containerization.

## Customization

- Add migration scripts in `backend/alembic/` or `backend/migrations/` as needed.
- Extend React components in `frontend/src/components/`.

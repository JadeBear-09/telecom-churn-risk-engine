FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .
RUN python -m src.train

EXPOSE 7860

CMD ["streamlit", "run", "dashboard/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]


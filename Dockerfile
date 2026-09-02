# Hugging Face Spaces runs this. Its free tier gives 16 GB of memory against
# Render's 512 MB, which matters here: importing pandas, scikit-learn and
# matplotlib together costs a few hundred megabytes before anything is trained.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    DJANGO_DEBUG=false

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Spaces mounts the code read-only in places and runs as a non-root user, so
# the directories written at runtime are created and handed over up front.
RUN mkdir -p media/project1 && \
    python manage.py collectstatic --no-input && \
    useradd -m app && chown -R app:app /app
USER app

ENV PORT=7860
EXPOSE 7860
CMD python manage.py migrate --no-input && \
    gunicorn pbl.wsgi:application --bind "0.0.0.0:$PORT" --workers 1 --timeout 120

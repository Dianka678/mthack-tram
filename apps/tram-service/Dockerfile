FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8080
WORKDIR /srv
COPY app/ ./app/
COPY data/ ./data/
COPY scripts/entrypoint.py ./entrypoint.py
RUN useradd --system --uid 10001 tram && chown -R tram:tram /srv
EXPOSE 8080
CMD ["python", "/srv/entrypoint.py"]

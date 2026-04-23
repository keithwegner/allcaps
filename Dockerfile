FROM node:22-slim AS frontend-build

WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend ./
RUN npm run build

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    ALLCAPS_RUNTIME=web \
    ALLCAPS_STATE_DIR=/var/data

RUN groupadd --system allcaps \
    && useradd --system --create-home --gid allcaps --shell /bin/bash allcaps

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt

COPY . /app
COPY --from=frontend-build /app/frontend/dist /app/frontend/dist

RUN chmod +x /app/scripts/start_app.sh /app/scripts/start_render.sh \
    && mkdir -p /var/data \
    && chown -R allcaps:allcaps /app /var/data

USER allcaps

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 CMD python -c "import os, sys, urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.environ.get('PORT', '8000')}\", timeout=3); sys.exit(0)"

CMD ["bash", "scripts/start_app.sh"]

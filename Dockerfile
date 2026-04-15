FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501 \
    ALLCAPS_STATE_DIR=/var/data

RUN groupadd --system allcaps \
    && useradd --system --create-home --gid allcaps --shell /bin/bash allcaps

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt

COPY . /app

RUN chmod +x /app/scripts/start_app.sh /app/scripts/start_render.sh \
    && mkdir -p /var/data \
    && chown -R allcaps:allcaps /app /var/data

USER allcaps

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 CMD python -c "import sys, urllib.request; urllib.request.urlopen('http://127.0.0.1:8501', timeout=3); sys.exit(0)"

CMD ["bash", "scripts/start_app.sh"]

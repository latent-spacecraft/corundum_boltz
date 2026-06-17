# Minimal RDKit ligand-blob preprocessing image.
#
# Build context MUST be packages/boltz (so the vendored core under
# scripts/python is reachable):
#
#   docker build -f server/Dockerfile -t corundum-blobber .   # run from packages/boltz
#
# Cloudflare Containers builds this for you when image_build_context is set to
# the parent dir (see server/wrangler.jsonc).
FROM python:3.12-slim

WORKDIR /app

# RDKit's manylinux wheel needs libXrender/libXext at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The RDKit-only core (single source of truth, shared with the CLI) + server.
COPY scripts/python/ligand_blob.py /app/ligand_blob.py
COPY server/app.py /app/app.py

ENV PORT=8080
EXPOSE 8080

# Liveness without an extra dep — hit the in-process health route.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz').read()" || exit 1

CMD ["python", "app.py"]

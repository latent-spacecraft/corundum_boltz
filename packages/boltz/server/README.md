# Ligand blob preprocessing server

The production half of "bring your own SMILES". Boltz needs an RDKit-derived
blob (heavy atoms + bonds + 6 geometry-constraint groups) for any non-CCD
ligand. RDKit's distance-geometry bounds + ETKDG embedding can't run in the
browser (`GetMoleculeBoundsMatrix` isn't in rdkit-js), so this tiny service
computes the blob (~tens of ms) and the heavy inference stays client-side.

It exposes the **same** route + contract as the Vite dev endpoint, so the SPA
(`loadLigandBlobFromSmiles` → `POST /__smiles_to_blob`) is deployment-agnostic.

```
POST /__smiles_to_blob   { "smiles": "...", "code"?: "ABC" }
  → 200 { ok: true, code, canonical_smiles, data: <blob> }
  → 422 { ok: false, error }            # bad SMILES / embedding failure
GET  /healthz → "ok"
```

## What's here

| file | role |
|------|------|
| `app.py` | stdlib HTTP server (no FastAPI) over the RDKit core |
| `../scripts/python/ligand_blob.py` | the RDKit-only core — **single source of truth**, shared with the CLI |
| `requirements.txt` | `rdkit` + `numpy` only (no torch/boltz) |
| `Dockerfile` | minimal image; build context = `packages/boltz` |
| `worker.js` + `wrangler.jsonc` | Cloudflare Worker routing `/__smiles_to_blob` to the container |

The core is byte-equivalent to Boltz's own extractors — proven by
`../scripts/python/verify_blob_equiv.py`. Re-run it if you bump RDKit.

## Local run (no Docker)

Any env with `rdkit` + `numpy`:

```bash
cd packages/boltz
PYTHONPATH=scripts/python PORT=8080 python server/app.py
curl -s localhost:8080/__smiles_to_blob -d '{"smiles":"c1ccccc1"}'
```

## Local run (Docker)

```bash
cd packages/boltz
docker build -f server/Dockerfile -t corundum-blobber .   # context = packages/boltz
docker run --rm -p 8080:8080 corundum-blobber
```

## Deploy to Cloudflare (Containers)

Requires the **Workers Paid** plan (Containers aren't on the free tier).

```bash
cd packages/boltz/server
npm install
npx wrangler deploy        # builds the Dockerfile, pushes the image, deploys the Worker
```

Then make the endpoint **same-origin** with your static site so the browser
fetch isn't cross-origin: in the Cloudflare dashboard (or wrangler `routes`),
add a route

```
yoursite.com/__smiles_to_blob*  →  corundum-blobber
```

(The Worker also sets permissive CORS headers, so a separate-origin
`*.workers.dev` deployment works too — but same-origin is cleaner and avoids a
preflight per request.)

### If `image_build_context` isn't supported by your wrangler

Build + push the image yourself and reference it directly:

```bash
cd packages/boltz
docker build -f server/Dockerfile -t <registry>/corundum-blobber:latest .
docker push <registry>/corundum-blobber:latest
# then in wrangler.jsonc replace  "image": "./Dockerfile", "image_build_context": ".."
#                          with   "image": "<registry>/corundum-blobber:latest"
```

## Not Cloudflare?

The container is a plain HTTP server — it runs anywhere (Fly.io, Cloud Run,
Render, a VM). Point the SPA at it by ensuring `/__smiles_to_blob` resolves on
the site's origin (reverse-proxy that path to the container).

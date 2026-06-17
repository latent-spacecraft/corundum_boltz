#!/usr/bin/env python3
"""
Minimal ligand-blob preprocessing server — the production half of "bring your
own SMILES". Pure stdlib HTTP (no FastAPI/uvicorn) over the RDKit-only core
(ligand_blob.py); the image only needs `rdkit` + `numpy`.

Routes:
  POST /__smiles_to_blob   { "smiles": "...", "code"?: "ABC" }
       → 200 { ok, code, canonical_smiles, data: <blob> }
       → 422 { ok: false, error }     (bad SMILES / embedding failure)
  GET  /healthz            → 200 "ok"

Same request/response contract as the Vite dev endpoint, so the client
(loadLigandBlobFromSmiles → POST /__smiles_to_blob) is deployment-agnostic.

Env: PORT (default 8080), MAX_SMILES_LEN (default 4000).
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import ligand_blob

PORT = int(os.environ.get("PORT", "8080"))
MAX_SMILES_LEN = int(os.environ.get("MAX_SMILES_LEN", "4000"))
ROUTE = "/__smiles_to_blob"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("/healthz", "/health"):
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?")[0].rstrip("/") != ROUTE:
            return self._send(404, {"ok": False, "error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 64 * 1024:
                return self._send(400, {"ok": False, "error": "missing or oversized body"})
            req = json.loads(self.rfile.read(length))
            smiles = req.get("smiles")
            code = req.get("code")
            if not isinstance(smiles, str) or not smiles.strip():
                return self._send(400, {"ok": False, "error": 'missing "smiles"'})
            if len(smiles) > MAX_SMILES_LEN:
                return self._send(400, {"ok": False, "error": f"SMILES too long (>{MAX_SMILES_LEN})"})
            if code is not None and not (isinstance(code, str) and code.isalnum() and len(code) <= 8):
                code = None  # ignore malformed override, fall back to hash
        except Exception as e:  # noqa: BLE001
            return self._send(400, {"ok": False, "error": f"bad request: {e}"})

        try:
            result = ligand_blob.build_from_smiles(smiles, code)
            self._send(200, {"ok": True, **result})
        except Exception as e:  # noqa: BLE001 — surface RDKit parse/embed errors cleanly
            self._send(422, {"ok": False, "error": str(e)})

    def log_message(self, fmt: str, *args) -> None:  # quieter logs
        sys.stderr.write("[blobber] " + (fmt % args) + "\n")


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    sys.stderr.write(f"[blobber] listening on :{PORT}{ROUTE}\n")
    server.serve_forever()


if __name__ == "__main__":
    main()

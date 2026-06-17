/**
 * Cloudflare Worker that fronts the RDKit blob container and exposes
 * POST /__smiles_to_blob (+ /healthz) — the production endpoint the SPA's
 * loadLigandBlobFromSmiles() calls.
 *
 * Deploy this on the SAME origin as the static site (add a route
 * `yoursite.com/__smiles_to_blob*` → this Worker) so the fetch is same-origin.
 * If you host the endpoint on a different origin, the permissive CORS headers
 * below let the isolated (COEP: credentialless) SPA reach it cross-origin.
 */
import { Container, getContainer } from '@cloudflare/containers'

export class BlobberContainer extends Container {
  defaultPort = 8080
  // Scale to zero between bursts; first request after idle pays a cold start.
  sleepAfter = '10m'
}

const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
  'Access-Control-Max-Age': '86400',
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url)
    const path = url.pathname.replace(/\/$/, '')

    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: CORS })
    }
    if (path !== '/__smiles_to_blob' && path !== '/healthz') {
      return new Response('not found', { status: 404 })
    }

    // One container instance handles all blob requests (stateless + cheap).
    // Bump getContainer's id / max_instances in wrangler.jsonc to shard.
    const container = getContainer(env.BLOBBER)
    const res = await container.fetch(request)
    const headers = new Headers(res.headers)
    for (const [k, v] of Object.entries(CORS)) headers.set(k, v)
    return new Response(res.body, { status: res.status, headers })
  },
}

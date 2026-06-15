/**
 * Streaming model fetcher.
 *
 * Downloads ONNX weights from a manifest URL, reports byte-level progress,
 * caches into OPFS, and returns the bytes for session creation. Models with
 * external-data sidecars (Boltz-2's graphs do — the bulk of the weights live
 * in `.onnx.data` next to a thin `.onnx`) get both files fetched and cached;
 * combined progress is reported as a single stream weighted by approxBytes,
 * so the UI shows one bar.
 *
 * Two robustness features sit under this surface, both in `fetchOne`:
 *
 *   1. Resume on disconnect. Each network segment is wrapped in a retry loop
 *      that re-issues `Range: bytes=N-` from the last byte already received,
 *      so a transient connection blip costs a brief pause and a re-handshake,
 *      not a restart from zero. Mid-session only — OPFS createWritable swaps
 *      atomically on close(), so a page refresh mid-download still loses the
 *      in-flight file. Cross-session resume needs SyncAccessHandle (worker-only).
 *
 *   2. Parallel chunked download. For files above PARALLEL_CHUNK_THRESHOLD,
 *      the body is split into PARALLEL_CHUNK_COUNT byte ranges fetched
 *      concurrently. HF's CDN, Cloudflare, and jsDelivr all honor Range and
 *      serve over HTTP/2, so 6 concurrent ranges hits a few-MB/s improvement
 *      on residential connections without saturating the per-origin pool.
 *      Each chunk has its own independent retry loop.
 *
 * If the server doesn't advertise Accept-Ranges, both features quietly
 * degrade to a single-shot streaming fetch — same behavior the old fetcher
 * had, just with a wrapping try/catch.
 */
import { getCache, type CacheKey } from './cache'
import {
  hasExternalData,
  totalApproxBytes,
  type ModelManifest,
} from './models/registry'

export type FetchPhase = 'check-cache' | 'downloading' | 'caching' | 'ready'

export interface FetchProgress {
  phase: FetchPhase
  bytesLoaded: number
  /** Total content length if known; otherwise undefined. */
  bytesTotal?: number
  /** Rough rate in bytes/sec, smoothed. */
  bytesPerSecond?: number
  /**
   * When a manifest has an external-data sidecar this is set so the UI can
   * label which file is currently flowing ('model' vs 'sidecar').
   */
  subject?: 'model' | 'sidecar'
}

export interface FetchOptions {
  signal?: AbortSignal
  onProgress?: (p: FetchProgress) => void
  /** Override cache tag (e.g., for content-hash-based keying). Defaults to manifest.id-derived tag. */
  tag?: string
  /** Force re-download even if a cached copy exists. */
  forceReload?: boolean
}

/** Bytes returned by the fetcher — sidecar present iff the manifest declared it. */
export interface FetchedModel {
  modelBytes: Uint8Array
  externalDataBytes?: Uint8Array
  externalDataFilename?: string
}

function deriveTag(manifest: ModelManifest): string {
  // URL host + path-tail acts as a passable identity tag when no explicit one
  // is supplied. The fetcher logs a warning when this is used — callers
  // really should specify a content hash.
  try {
    const u = new URL(manifest.url, globalThis.location?.href ?? 'http://localhost/')
    return `${u.host}_${u.pathname.replace(/[^a-z0-9.]/gi, '_').slice(-40)}`
  } catch {
    return manifest.url.replace(/[^a-z0-9.]/gi, '_').slice(-40)
  }
}

const SMOOTHING = 0.25 // EMA factor for bytes/sec

// Files at or above this size are fetched as PARALLEL_CHUNK_COUNT ranges.
// Below the threshold the per-chunk overhead (extra HEAD + connection setup)
// outweighs the parallel-throughput win.
const PARALLEL_CHUNK_THRESHOLD = 32 * 1024 * 1024
const PARALLEL_CHUNK_COUNT = 6

// Per-segment retry budget. 6 attempts × exponential backoff caps at ~24 s
// of cumulative wait before surfacing a hard failure to the orchestrator.
const MAX_ATTEMPTS_PER_SEGMENT = 6
const INITIAL_BACKOFF_MS = 500
const MAX_BACKOFF_MS = 8000

interface FetchOneOptions {
  url: string
  approxBytes: number
  cacheKey: CacheKey
  signal?: AbortSignal
  /**
   * Called whenever this single download makes progress; ALL counters are
   * relative to this file (not the combined total). The orchestrator
   * composes a combined progress object before forwarding to the user.
   */
  onLocalProgress?: (loaded: number, total: number | undefined, bps: number) => void
  forceReload: boolean
}

interface ProbeResult {
  size?: number
  ranges: boolean
}

/**
 * HEAD the URL to learn its Content-Length and whether the server honors
 * Range requests. Both HF (Xet & LFS), Cloudflare, and jsDelivr advertise
 * `Accept-Ranges: bytes`. If HEAD fails for any reason, we fall back to a
 * conservative single-shot fetch with no resume.
 */
async function probe(url: string, signal: AbortSignal | undefined): Promise<ProbeResult> {
  try {
    const res = await fetch(url, { method: 'HEAD', signal })
    if (!res.ok) return { ranges: false }
    const cl = res.headers.get('Content-Length')
    const ar = res.headers.get('Accept-Ranges')
    return {
      size: cl ? Number(cl) : undefined,
      ranges: ar?.toLowerCase() === 'bytes',
    }
  } catch {
    return { ranges: false }
  }
}

function backoffMs(attempt: number): number {
  return Math.min(MAX_BACKOFF_MS, INITIAL_BACKOFF_MS * 2 ** (attempt - 1))
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(signal.reason ?? new Error('aborted'))
    const t = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort)
      resolve()
    }, ms)
    const onAbort = () => {
      clearTimeout(t)
      reject(signal!.reason ?? new Error('aborted'))
    }
    signal?.addEventListener('abort', onAbort, { once: true })
  })
}

interface SegmentOptions {
  url: string
  /** Inclusive start byte for the segment within the full resource. */
  start: number
  /** Inclusive end byte, or undefined for "to end of resource". */
  end?: number
  /** Whether the server honors Range. When false, only [0, undefined] is valid. */
  useRanges: boolean
  signal?: AbortSignal
  /**
   * Called with a positive delta whenever new bytes arrive, or a negative
   * delta when a non-resumable retry rolls back accumulated progress.
   */
  onProgress: (delta: number) => void
}

/**
 * Fetch one byte range of one URL with retry-and-resume. If the connection
 * drops mid-stream, the next attempt issues `Range: bytes=(start+downloaded)-end`
 * so we pick up where we left off. If the server doesn't honor Range, retries
 * restart from byte 0 of the segment and the progress delta is rolled back.
 */
async function fetchSegment(opts: SegmentOptions): Promise<Uint8Array> {
  let downloaded = 0
  const chunks: Uint8Array[] = []
  let attempt = 0

  while (true) {
    attempt++
    try {
      const headers: Record<string, string> = {}
      if (opts.useRanges) {
        const rangeStart = opts.start + downloaded
        const rangeEnd = opts.end ?? ''
        headers['Range'] = `bytes=${rangeStart}-${rangeEnd}`
      } else if (downloaded > 0) {
        // Server doesn't support resume. Restart the segment from zero and
        // roll progress back so the combined UI counter stays truthful.
        opts.onProgress(-downloaded)
        chunks.length = 0
        downloaded = 0
      }

      const res = await fetch(opts.url, { headers, signal: opts.signal })
      const expectedStatus = opts.useRanges ? 206 : 200
      if (!res.ok || res.status !== expectedStatus) {
        // 200 in response to a Range request means the server ignored our
        // header (some CDNs do this for tiny files); accept it only if we
        // were starting from offset 0 and asked for the full remainder.
        const ok200OnRange =
          opts.useRanges && res.status === 200 && downloaded === 0 && opts.start === 0
        if (!ok200OnRange) {
          throw new Error(`HTTP ${res.status} ${res.statusText} from ${opts.url}`)
        }
      }

      if (!res.body) {
        const buf = new Uint8Array(await res.arrayBuffer())
        chunks.push(buf)
        downloaded += buf.byteLength
        opts.onProgress(buf.byteLength)
        return assemble(chunks, downloaded)
      }

      const reader = res.body.getReader()
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        if (!value) continue
        chunks.push(value)
        downloaded += value.byteLength
        opts.onProgress(value.byteLength)
      }
      return assemble(chunks, downloaded)
    } catch (err) {
      if (opts.signal?.aborted) throw err
      if (attempt >= MAX_ATTEMPTS_PER_SEGMENT) throw err
      // Brief pause, then loop and either resume (Range) or restart.
      await sleep(backoffMs(attempt), opts.signal)
    }
  }
}

function assemble(parts: Uint8Array[], total: number): Uint8Array {
  if (parts.length === 1 && parts[0].byteLength === total) return parts[0]
  const out = new Uint8Array(total)
  let off = 0
  for (const c of parts) {
    out.set(c, off)
    off += c.byteLength
  }
  return out
}

/**
 * Stream-fetch one URL, cache to OPFS, return the bytes.
 *
 * If the resource is large and the server supports ranges, the body is
 * split into PARALLEL_CHUNK_COUNT concurrent segment fetches; otherwise
 * we fetch it in one streamed segment. Either path retries with byte-Range
 * resume on transient errors.
 */
async function fetchOne(opts: FetchOneOptions): Promise<Uint8Array> {
  const cache = await getCache()
  if (!opts.forceReload && cache.available && (await cache.has(opts.cacheKey))) {
    const cached = await cache.read(opts.cacheKey)
    if (cached) {
      opts.onLocalProgress?.(cached.length, cached.length, 0)
      return cached
    }
  }

  const probed = await probe(opts.url, opts.signal)
  const total = probed.size ?? (opts.approxBytes > 0 ? opts.approxBytes : undefined)

  // Combined progress bookkeeping. Each segment posts +N (or -N on rollback)
  // and we re-emit the aggregate to the orchestrator on a 250 ms tick.
  let bytesLoaded = 0
  let lastTick = performance.now()
  let lastLoaded = 0
  let bps = 0
  const reportProgress = (force = false) => {
    const now = performance.now()
    const dt = now - lastTick
    if (!force && dt < 250) return
    const instantBps = dt > 0 ? ((bytesLoaded - lastLoaded) * 1000) / dt : 0
    if (instantBps > 0) {
      bps = bps === 0 ? instantBps : bps * (1 - SMOOTHING) + instantBps * SMOOTHING
    }
    lastTick = now
    lastLoaded = bytesLoaded
    opts.onLocalProgress?.(bytesLoaded, total, bps)
  }
  const onSegmentProgress = (delta: number) => {
    bytesLoaded = Math.max(0, bytesLoaded + delta)
    reportProgress()
  }

  // Decide single vs. parallel. We need a known total and Range support to
  // split; everything else degrades to one streamed segment with retry.
  const goParallel =
    probed.ranges && total !== undefined && total >= PARALLEL_CHUNK_THRESHOLD

  let bytes: Uint8Array
  // Bind cancellation: if any segment fails terminally, abort the rest so
  // they stop burning bandwidth in the background.
  const ctrl = new AbortController()
  const onParentAbort = () => ctrl.abort(opts.signal?.reason)
  opts.signal?.addEventListener('abort', onParentAbort, { once: true })

  try {
    if (goParallel) {
      const ranges = splitRanges(total!, PARALLEL_CHUNK_COUNT)
      const segments = await Promise.all(
        ranges.map((r) =>
          fetchSegment({
            url: opts.url,
            start: r.start,
            end: r.end,
            useRanges: true,
            signal: ctrl.signal,
            onProgress: onSegmentProgress,
          }),
        ),
      )
      bytes = assemble(segments, total!)
    } else {
      bytes = await fetchSegment({
        url: opts.url,
        start: 0,
        end: total !== undefined ? total - 1 : undefined,
        useRanges: probed.ranges && total !== undefined,
        signal: ctrl.signal,
        onProgress: onSegmentProgress,
      })
    }
  } catch (err) {
    ctrl.abort()
    throw err
  } finally {
    opts.signal?.removeEventListener('abort', onParentAbort)
  }

  // Final progress emit so the bar lands cleanly at 100%.
  bytesLoaded = bytes.length
  reportProgress(true)

  // Write to OPFS in one shot. The old fetcher streamed to OPFS as bytes
  // arrived, but createWritable swaps atomically on close() anyway — there's
  // no durability win mid-stream. One write + close keeps the code clean.
  if (cache.available) {
    try {
      const stream = await cache.writeStream(opts.cacheKey)
      await stream.write(bytes)
      await stream.close()
    } catch {
      await cache.evict(opts.cacheKey).catch(() => undefined)
    }
  }

  return bytes
}

interface RangePlan {
  start: number
  end: number
}

function splitRanges(total: number, count: number): RangePlan[] {
  const chunkSize = Math.ceil(total / count)
  const out: RangePlan[] = []
  for (let i = 0; i < count; i++) {
    const start = i * chunkSize
    if (start >= total) break
    const end = Math.min(total - 1, start + chunkSize - 1)
    out.push({ start, end })
  }
  return out
}

/**
 * Fetch (or load-from-cache) all bytes a manifest needs to create an ORT
 * session: the .onnx file, and (if declared) its .onnx.data sidecar.
 */
export async function fetchModel(
  manifest: ModelManifest,
  opts: FetchOptions = {},
): Promise<FetchedModel> {
  const tag = opts.tag ?? deriveTag(manifest)
  const emit = (p: FetchProgress) => opts.onProgress?.(p)

  emit({ phase: 'check-cache', bytesLoaded: 0 })

  const totalCombined = totalApproxBytes(manifest)
  let modelBytesLoaded = 0
  let sidecarBytesLoaded = 0
  let lastEmittedBps = 0

  // Compose a combined progress emit. Called from each per-file callback;
  // both counters live in closure so the running sum is always correct.
  const emitCombined = (
    subject: 'model' | 'sidecar',
    phase: FetchPhase,
    bps: number,
  ) => {
    if (bps > 0) lastEmittedBps = bps
    const loaded = modelBytesLoaded + sidecarBytesLoaded
    emit({
      phase,
      bytesLoaded: loaded,
      bytesTotal: totalCombined,
      bytesPerSecond: lastEmittedBps,
      subject,
    })
  }

  // 1. The .onnx file itself.
  const modelBytes = await fetchOne({
    url: manifest.url,
    approxBytes: manifest.approxBytes,
    cacheKey: { modelId: manifest.id, tag: `${tag}__onnx` },
    signal: opts.signal,
    forceReload: opts.forceReload === true,
    onLocalProgress: (loaded, _total, bps) => {
      modelBytesLoaded = loaded
      emitCombined('model', 'downloading', bps)
    },
  })
  modelBytesLoaded = modelBytes.length

  // 2. The .onnx.data sidecar, if any.
  let externalDataBytes: Uint8Array | undefined
  if (hasExternalData(manifest)) {
    externalDataBytes = await fetchOne({
      url: manifest.externalDataUrl!,
      approxBytes: manifest.externalDataApproxBytes ?? 0,
      cacheKey: { modelId: manifest.id, tag: `${tag}__data` },
      signal: opts.signal,
      forceReload: opts.forceReload === true,
      onLocalProgress: (loaded, _total, bps) => {
        sidecarBytesLoaded = loaded
        emitCombined('sidecar', 'downloading', bps)
      },
    })
    sidecarBytesLoaded = externalDataBytes.length
  }

  emit({
    phase: 'ready',
    bytesLoaded: modelBytesLoaded + sidecarBytesLoaded,
    bytesTotal: totalCombined,
    bytesPerSecond: lastEmittedBps,
  })

  return {
    modelBytes,
    externalDataBytes,
    externalDataFilename: externalDataBytes ? manifest.externalDataFilename : undefined,
  }
}

export function formatBytes(n: number): string {
  if (!Number.isFinite(n)) return '—'
  const units = ['B', 'kB', 'MB', 'GB', 'TB']
  let v = n
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v.toFixed(v < 10 ? 2 : v < 100 ? 1 : 0)} ${units[i]}`
}

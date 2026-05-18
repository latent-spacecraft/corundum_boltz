/**
 * OPFS-backed model cache.
 *
 * Models can be multiple gigabytes. The Origin Private File System gives us a
 * sandboxed, persistent, fast filesystem inside the browser — perfect for
 * cached weights. The API is deliberately small: read / writeStream / has /
 * size / list / evict. Acts never touch this directly; the fetcher does.
 *
 * Cache key shape: `${modelId}__${sha256OrTag}`. Including the content tag in
 * the filename means an upgraded model just lands beside the old one and the
 * old one can be evicted by id when we're sure no session still references
 * its bytes.
 *
 * If OPFS is unavailable (older browsers, some private modes) the API falls
 * back to a no-op store. The fetcher will then re-download every session.
 */

const CACHE_DIR = 'corundum-weights'

interface FileSystemDirectoryHandleLike {
  getDirectoryHandle: (
    name: string,
    options?: { create?: boolean },
  ) => Promise<FileSystemDirectoryHandleLike>
  getFileHandle: (
    name: string,
    options?: { create?: boolean },
  ) => Promise<FileSystemFileHandleLike>
  removeEntry: (name: string, options?: { recursive?: boolean }) => Promise<void>
  values: () => AsyncIterable<FileSystemHandleLike>
}
interface FileSystemFileHandleLike {
  kind: 'file'
  name: string
  getFile: () => Promise<File>
  createWritable: (options?: {
    keepExistingData?: boolean
  }) => Promise<FileSystemWritableFileStreamLike>
}
interface FileSystemHandleLike {
  kind: 'file' | 'directory'
  name: string
}
interface FileSystemWritableFileStreamLike extends WritableStream<Uint8Array> {
  write: (chunk: Uint8Array | { type: 'write'; data: Uint8Array }) => Promise<void>
  close: () => Promise<void>
}

async function rootDir(): Promise<FileSystemDirectoryHandleLike | null> {
  const storage = (navigator as unknown as {
    storage?: { getDirectory?: () => Promise<FileSystemDirectoryHandleLike> }
  }).storage
  if (!storage?.getDirectory) return null
  try {
    const root = await storage.getDirectory()
    return root.getDirectoryHandle(CACHE_DIR, { create: true })
  } catch {
    return null
  }
}

export interface CacheKey {
  modelId: string
  /** Content tag (sha256, etag, or version string). Forms part of the filename. */
  tag: string
}

function fileNameFor(key: CacheKey): string {
  // Slashes are illegal in OPFS filenames; replace with __
  const safeId = key.modelId.replace(/[/\\]/g, '__')
  const safeTag = key.tag.replace(/[/\\]/g, '__')
  return `${safeId}__${safeTag}.onnx`
}

export interface CacheEntry {
  key: CacheKey
  fileName: string
  bytes: number
  lastModified: number
}

export interface ModelCache {
  available: boolean
  has(key: CacheKey): Promise<boolean>
  read(key: CacheKey): Promise<Uint8Array | null>
  /** Returns a writable stream that finalises the cache entry on close(). */
  writeStream(key: CacheKey): Promise<FileSystemWritableFileStreamLike>
  evict(key: CacheKey): Promise<void>
  list(): Promise<CacheEntry[]>
  totalBytes(): Promise<number>
}

class NoopCache implements ModelCache {
  available = false
  async has() {
    return false
  }
  async read() {
    return null
  }
  async writeStream(): Promise<FileSystemWritableFileStreamLike> {
    throw new Error('OPFS unavailable: cache is read-only-no-op in this environment')
  }
  async evict() {}
  async list() {
    return []
  }
  async totalBytes() {
    return 0
  }
}

class OpfsCache implements ModelCache {
  available = true
  private dir: FileSystemDirectoryHandleLike
  constructor(dir: FileSystemDirectoryHandleLike) {
    this.dir = dir
  }

  async has(key: CacheKey): Promise<boolean> {
    try {
      await this.dir.getFileHandle(fileNameFor(key))
      return true
    } catch {
      return false
    }
  }

  async read(key: CacheKey): Promise<Uint8Array | null> {
    try {
      const handle = await this.dir.getFileHandle(fileNameFor(key))
      const file = await handle.getFile()
      return new Uint8Array(await file.arrayBuffer())
    } catch {
      return null
    }
  }

  async writeStream(key: CacheKey): Promise<FileSystemWritableFileStreamLike> {
    const handle = await this.dir.getFileHandle(fileNameFor(key), { create: true })
    return handle.createWritable({ keepExistingData: false })
  }

  async evict(key: CacheKey): Promise<void> {
    try {
      await this.dir.removeEntry(fileNameFor(key))
    } catch {
      /* missing entry is fine */
    }
  }

  async list(): Promise<CacheEntry[]> {
    const out: CacheEntry[] = []
    for await (const handle of this.dir.values()) {
      if (handle.kind !== 'file') continue
      const fileHandle = handle as unknown as FileSystemFileHandleLike
      const file = await fileHandle.getFile()
      const parts = handle.name.replace(/\.onnx$/, '').split('__')
      if (parts.length < 2) continue
      const tag = parts.pop()!
      const modelId = parts.join('__')
      out.push({
        key: { modelId, tag },
        fileName: handle.name,
        bytes: file.size,
        lastModified: file.lastModified,
      })
    }
    return out
  }

  async totalBytes(): Promise<number> {
    const entries = await this.list()
    return entries.reduce((acc, e) => acc + e.bytes, 0)
  }
}

let cached: ModelCache | undefined
export async function getCache(): Promise<ModelCache> {
  if (cached) return cached
  const dir = await rootDir()
  cached = dir ? new OpfsCache(dir) : new NoopCache()
  return cached
}

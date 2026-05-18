/**
 * Loader for the binary feats blob produced by
 * `boltz-dev/scripts/feats_to_blob.py`.
 *
 * Blob format:
 *   bytes 0..3   : uint32 LE = header length N
 *   bytes 4..4+N : UTF-8 JSON header
 *     {
 *       schema_version: '0.1',
 *       tensors: [{name, shape, dtype, offset, byteLength}, ...],
 *       B, N, A, K
 *     }
 *   bytes 4+N..  : raw tensor bytes, in the order listed in `tensors`
 *
 * dtype → typed-array constructor:
 *   float32 → Float32Array
 *   float16 → Uint16Array (raw fp16 bits)
 *   int64   → BigInt64Array
 *   int32   → Int32Array
 *   int16   → Int16Array
 *   int8    → Int8Array
 *   uint8   → Uint8Array
 *   bool    → Uint8Array  (0 or 1)
 */

export type FeatsDtype =
  | 'float32'
  | 'float16'
  | 'int64'
  | 'int32'
  | 'int16'
  | 'int8'
  | 'uint8'
  | 'bool'

export interface FeatsTensor {
  name: string
  shape: number[]
  dtype: FeatsDtype
  data:
    | Float32Array
    | Uint16Array
    | BigInt64Array
    | Int32Array
    | Int16Array
    | Int8Array
    | Uint8Array
}

export interface FeatsBundle {
  schemaVersion: string
  B: number
  N: number
  A: number
  K: number
  tensors: Record<string, FeatsTensor>
}

interface RawDescriptor {
  name: string
  shape: number[]
  dtype: FeatsDtype
  offset: number
  byteLength: number
}

interface RawHeader {
  schema_version: string
  tensors: RawDescriptor[]
  B: number
  N: number
  A: number
  K: number
}

function viewFor(dtype: FeatsDtype, buf: ArrayBuffer, offset: number, byteLength: number): FeatsTensor['data'] {
  // Slice into a fresh ArrayBuffer so the typed-array view starts at byte 0
  // and is naturally aligned, regardless of where the tensor sits in the
  // concatenated blob. (Tensors are packed back-to-back; a 1-byte bool
  // tensor of odd length leaves the next tensor at a misaligned offset
  // that Float32Array / BigInt64Array constructors reject.)
  const sliced = buf.slice(offset, offset + byteLength)
  switch (dtype) {
    case 'float32':
      return new Float32Array(sliced)
    case 'float16':
      return new Uint16Array(sliced)
    case 'int64':
      return new BigInt64Array(sliced)
    case 'int32':
      return new Int32Array(sliced)
    case 'int16':
      return new Int16Array(sliced)
    case 'int8':
      return new Int8Array(sliced)
    case 'uint8':
    case 'bool':
      return new Uint8Array(sliced)
    default: {
      const exhaustive: never = dtype
      throw new Error(`Unsupported dtype: ${exhaustive as string}`)
    }
  }
}

export async function fetchFeats(url: string): Promise<FeatsBundle> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Feats fetch failed: ${res.status} ${res.statusText}`)
  const buf = await res.arrayBuffer()
  return parseFeats(buf)
}

export function parseFeats(buf: ArrayBuffer): FeatsBundle {
  const dv = new DataView(buf)
  const headerLen = dv.getUint32(0, true /* little-endian */)
  const headerBytes = new Uint8Array(buf, 4, headerLen)
  const headerText = new TextDecoder().decode(headerBytes)
  const header = JSON.parse(headerText) as RawHeader
  const bodyOffset = 4 + headerLen
  const tensors: Record<string, FeatsTensor> = {}
  for (const desc of header.tensors) {
    const data = viewFor(desc.dtype, buf, bodyOffset + desc.offset, desc.byteLength)
    tensors[desc.name] = {
      name: desc.name,
      shape: desc.shape,
      dtype: desc.dtype,
      data,
    }
  }
  return {
    schemaVersion: header.schema_version,
    B: header.B,
    N: header.N,
    A: header.A,
    K: header.K,
    tensors,
  }
}

/** Get a tensor, throwing a useful error if it's missing. */
export function tensor(bundle: FeatsBundle, name: string): FeatsTensor {
  const t = bundle.tensors[name]
  if (!t) {
    const have = Object.keys(bundle.tensors).slice(0, 8).join(', ')
    throw new Error(`feats[${name}] missing — first few present: ${have}…`)
  }
  return t
}

/**
 * Memory pressure estimator for the FASTA input box.
 *
 * Goal: live, color-coded warning as the user pastes / types sequences,
 * so they can see when they're about to ask the browser to do something
 * it can't.
 *
 * Model (kept deliberately coarse — alpha targets calibration, not
 * accuracy):
 *
 *   total ≈ weights(precision) + activations(N)
 *   activations(N) ≈ k · N² · 4 · token_z  +  c · N · 4 · token_s
 *
 * The trunk's pair representation z [1, N, N, 128] in fp32 dominates;
 * empirically ORT holds ~6 simultaneous N²-shaped intermediates during
 * trunk forward (pair updates, attention scores, the running z), so we
 * use a 6× fudge on the z footprint.
 *
 * Available memory: navigator.deviceMemory (capped 8 GB in Chrome,
 * absent elsewhere) halved to leave the OS, Mol\*, and other tabs some
 * room. Falls back to a 2 GB assumption when unknown.
 *
 * N estimate: cheap line-pass over the FASTA — sum sequence lengths,
 * +50 per ligand chain (typical heavy-atom count). Not exact; good
 * enough to pick a color.
 */
import type { BoltzPrecision } from '@/acts/boltz/models'
import { bundleApproxBytes } from '@/acts/boltz/models'
import type { DeviceCapabilities, DeviceTier } from './device'

const TOKEN_S = 384
const TOKEN_Z = 128
const ACTIVATION_FUDGE = 6
const LIGAND_ATOM_GUESS = 50

/**
 * Total token count estimate from a raw FASTA string. Doesn't run the
 * full parser — we don't need chain validity for a memory bar.
 */
export function estimateN(fasta: string): number {
  if (!fasta.trim()) return 0
  const lines = fasta.split('\n')
  let n = 0
  let curType: 'polymer' | 'ligand' = 'polymer'
  let curSeq = ''
  const flush = () => {
    if (!curSeq && curType === 'polymer') return
    if (curType === 'ligand') {
      n += LIGAND_ATOM_GUESS
    } else {
      n += curSeq.replace(/\s+/g, '').length
    }
    curSeq = ''
  }
  for (const line of lines) {
    if (line.startsWith('>')) {
      flush()
      curType = /\bligand\b/i.test(line) ? 'ligand' : 'polymer'
    } else {
      curSeq += line
    }
  }
  flush()
  return n
}

export type PressureLevel = 'idle' | 'green' | 'yellow' | 'red'

export interface MemoryEstimate {
  /** Estimated token count (residues + per-atom ligand contributions). */
  n: number
  /** Bundle weights (fixed per precision). */
  weightsBytes: number
  /** Estimated peak activation footprint, dominated by N². */
  activationsBytes: number
  /** weights + activations. */
  totalBytes: number
  /** What we think the page has to work with. */
  availableBytes: number
  /** total / available. >1 means projected to OOM. */
  pressureRatio: number
  level: PressureLevel
  /** Short, user-facing summary line. */
  reason: string
}

/**
 * Per-tier fallback assumption for total system RAM when
 * `navigator.deviceMemory` is unavailable (Safari, Firefox, and many
 * macOS Chrome builds). These are deliberately conservative — they
 * represent a realistic *floor* for the tier, not the median.
 *
 *   mobile              :  6 GB  (modern phones land 6-12, tablets higher)
 *   desktop-integrated  :  8 GB  (low-end laptop with shared GPU)
 *   desktop-discrete    : 16 GB  (anything with a discrete GPU/Apple Silicon)
 */
const TIER_RAM_FLOOR_GB: Record<DeviceTier, number> = {
  mobile: 6,
  'desktop-integrated': 8,
  'desktop-discrete': 16,
}

/** Fraction of system RAM we assume is available to the page (the rest
 *  goes to OS + browser chrome + other tabs + Mol\*). */
const FREE_FRACTION = 0.5

export function estimateMemory(
  fasta: string,
  precision: BoltzPrecision,
  device: DeviceCapabilities | null,
): MemoryEstimate {
  const n = estimateN(fasta)
  const weightsBytes = bundleApproxBytes(precision)
  const zBytes = n * n * TOKEN_Z * 4
  const sBytes = n * TOKEN_S * 4
  const activationsBytes = ACTIVATION_FUDGE * zBytes + 4 * sBytes
  const totalBytes = weightsBytes + activationsBytes

  // Available memory: prefer the browser-reported value (Chrome only,
  // capped at 8 GB for privacy). Fall back to a tier-aware floor —
  // crucial on Safari/Firefox/macOS-Chrome where deviceMemory is
  // undefined and a flat 2 GB assumption made fp32 weights alone
  // OOM on devices with 32+ GB free.
  const ramGB =
    device?.deviceMemoryGB ??
    (device ? TIER_RAM_FLOOR_GB[device.tier] : TIER_RAM_FLOOR_GB.mobile)
  const availableBytes = ramGB * 1024 ** 3 * FREE_FRACTION

  const pressureRatio = totalBytes / availableBytes

  let level: PressureLevel
  if (n === 0) level = 'idle'
  else if (pressureRatio < 0.5) level = 'green'
  else if (pressureRatio < 0.9) level = 'yellow'
  else level = 'red'

  const reason =
    n === 0
      ? 'No sequence yet.'
      : level === 'red'
        ? `~${n} tokens projected to exhaust available memory — expect OOM.`
        : level === 'yellow'
          ? `~${n} tokens — tight, expect long stalls during recycling.`
          : `~${n} tokens — comfortable.`

  return {
    n,
    weightsBytes,
    activationsBytes,
    totalBytes,
    availableBytes,
    pressureRatio,
    level,
    reason,
  }
}

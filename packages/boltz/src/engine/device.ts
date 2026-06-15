/**
 * Device → precision tier autodetect.
 *
 * Probes the browser for the signals that actually determine which Boltz-2
 * precision the device can run comfortably:
 *
 *   - **WebGPU adapter** (the floor): no adapter → CPU/WASM only → int8
 *   - **navigator.deviceMemory** (Chrome only, capped at 8): coarse RAM hint
 *   - **userAgentData.mobile** (modern UA-CH) or UA regex (fallback):
 *     phones and tablets always route to int8 regardless of GPU
 *
 * Tiers map to one of the visible precisions in `PRECISIONS` (`models.ts`).
 * fp16 is intentionally absent from that list for the alpha (no JS pack/
 * unpack at graph boundaries yet); when it returns the recommendation table
 * here gets a mid-tier slot too.
 *
 * Cheap to call: one async adapter probe, otherwise sync. Returns a single
 * snapshot — call once at mount, don't re-poll.
 */
import type { BoltzPrecision } from '@/acts/boltz/models'

export type DeviceTier = 'mobile' | 'desktop-integrated' | 'desktop-discrete'

export interface DeviceCapabilities {
  tier: DeviceTier
  webgpu: boolean
  /** Adapter vendor/description if WebGPU was reachable, else undefined. */
  adapterLabel?: string
  /** Chrome's deviceMemory (GB, capped at 8). Undefined on non-Chromium. */
  deviceMemoryGB?: number
  /** True if UA-CH or UA regex says this is a phone/tablet. */
  isMobile: boolean
  /** Precision the picker should pre-select on first paint. */
  recommendedPrecision: BoltzPrecision
  /** Short, user-facing line explaining the recommendation. */
  reason: string
}

interface NavigatorUaDataExtras {
  userAgentData?: { mobile?: boolean; platform?: string }
  deviceMemory?: number
}

interface AdapterInfoLike {
  vendor?: string
  description?: string
  architecture?: string
}

// Older Chromium exposed adapter info only via the async requestAdapterInfo()
// method; current builds put it on `adapter.info`. We type-narrow off both.
interface AdapterWithInfo {
  requestAdapterInfo?(): Promise<AdapterInfoLike>
  info?: AdapterInfoLike
}

async function probeWebGPU(): Promise<{ ok: boolean; label?: string }> {
  const gpu = navigator.gpu as unknown as
    | { requestAdapter(opts?: { powerPreference?: 'high-performance' | 'low-power' }): Promise<unknown> }
    | undefined
  if (!gpu) return { ok: false }
  try {
    const adapter = (await gpu.requestAdapter({
      powerPreference: 'high-performance',
    })) as AdapterWithInfo | null
    if (!adapter) return { ok: false }
    const info =
      adapter.info ??
      (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : undefined)
    const label =
      info?.description?.trim() ||
      info?.vendor?.trim() ||
      info?.architecture?.trim() ||
      undefined
    return { ok: true, label }
  } catch {
    return { ok: false }
  }
}

function probeMobile(): boolean {
  const nav = navigator as Navigator & NavigatorUaDataExtras
  if (typeof nav.userAgentData?.mobile === 'boolean') {
    return nav.userAgentData.mobile
  }
  // UA fallback. iPadOS lies about being macOS but exposes touch points;
  // we treat any iPad-class device as mobile.
  const ua = navigator.userAgent
  if (/Mobi|Android|iPhone|iPod/.test(ua)) return true
  if (/iPad/.test(ua)) return true
  // Newer iPads identify as Mac. Detect by touch + Mac UA.
  if (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1) return true
  return false
}

function probeMemory(): number | undefined {
  const nav = navigator as Navigator & NavigatorUaDataExtras
  return typeof nav.deviceMemory === 'number' ? nav.deviceMemory : undefined
}

export async function detectDevice(): Promise<DeviceCapabilities> {
  const [{ ok: webgpu, label: adapterLabel }, isMobile] = [
    await probeWebGPU(),
    probeMobile(),
  ]
  const deviceMemoryGB = probeMemory()

  // ── Classify ───────────────────────────────────────────────────────────
  let tier: DeviceTier
  if (isMobile || !webgpu) {
    tier = 'mobile'
  } else if (deviceMemoryGB !== undefined && deviceMemoryGB < 8) {
    // Chrome caps deviceMemory at 8 — `< 8` is "definitely < 8 GB", treat
    // as integrated/low. `>= 8` is "≥ 8 GB or unknown-capable", treat as
    // discrete-tier (the picker will still let them downgrade).
    tier = 'desktop-integrated'
  } else {
    tier = 'desktop-discrete'
  }

  // ── Recommend ──────────────────────────────────────────────────────────
  // fp16 is hidden for the alpha (see models.ts), so the recommendation
  // collapses to a binary: int8 for anything below the desktop-discrete
  // line, fp32 for high-tier desktops where 2 GB of weights and PyTorch-
  // parity precision is the right trade. The user can always override.
  let recommendedPrecision: BoltzPrecision
  let reason: string
  if (tier === 'desktop-discrete') {
    recommendedPrecision = 'fp32'
    reason = adapterLabel
      ? `${adapterLabel} + ≥ 8 GB RAM — fp32 reference precision recommended.`
      : 'WebGPU + ≥ 8 GB RAM — fp32 reference precision recommended.'
  } else if (tier === 'desktop-integrated') {
    recommendedPrecision = 'int8'
    reason = 'WebGPU detected but < 8 GB RAM — int8 keeps the download light.'
  } else if (isMobile) {
    recommendedPrecision = 'int8'
    reason = 'Mobile device — int8 routes through WASM+SIMD, no WebGPU dependency.'
  } else {
    recommendedPrecision = 'int8'
    reason = 'No WebGPU adapter — int8 (WASM) is the only viable tier here.'
  }

  return {
    tier,
    webgpu,
    adapterLabel,
    deviceMemoryGB,
    isMobile,
    recommendedPrecision,
    reason,
  }
}

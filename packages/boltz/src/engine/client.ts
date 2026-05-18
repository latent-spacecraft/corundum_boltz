/**
 * Engine client — the main-thread façade for the worker-hosted engine.
 *
 * One worker, one singleton proxy. Acts import { engine } from '@/engine/client'
 * and never touch Comlink or workers directly.
 */
import * as Comlink from 'comlink'
import type { EngineWorkerApi } from './worker'

let proxy: Comlink.Remote<EngineWorkerApi> | null = null
let worker: Worker | null = null

function spawn(): Comlink.Remote<EngineWorkerApi> {
  if (proxy) return proxy
  worker = new Worker(new URL('./worker.ts', import.meta.url), {
    type: 'module',
    name: 'corundum-engine',
  })
  proxy = Comlink.wrap<EngineWorkerApi>(worker)
  return proxy
}

/**
 * Engine API. Lazily instantiates the worker on first access.
 * All methods are async (Comlink proxies them across the boundary).
 */
export const engine = new Proxy({} as Comlink.Remote<EngineWorkerApi>, {
  get(_t, prop) {
    const p = spawn()
    return Reflect.get(p, prop)
  },
})

/** Wrap a callback so it can be invoked across the worker boundary. */
export function proxyCallback<T extends (...args: never[]) => unknown>(fn: T): T {
  return Comlink.proxy(fn) as unknown as T
}

/** Terminate the worker. Useful for tests; the page rarely needs it. */
export function shutdown() {
  if (worker) {
    worker.terminate()
    worker = null
    proxy = null
  }
}

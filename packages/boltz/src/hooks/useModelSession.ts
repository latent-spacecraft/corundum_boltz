/**
 * useModelSession — the act-side hook for living with a model.
 *
 * Given a manifest, returns a small state machine plus the methods to drive
 * it. Tracks fetch progress and surfaces it in a form the UI can render
 * directly. Disposes the worker session on unmount.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { engine, proxyCallback } from '@/engine/client'
import type { FetchProgress } from '@/engine/fetcher'
import type { ModelManifest } from '@/engine/models/registry'
import type { FeedDict, SessionHandle, TensorPayload } from '@/engine/worker'

export type SessionStatus =
  | 'idle'
  | 'fetching'
  | 'compiling'
  | 'ready'
  | 'running'
  | 'error'

export interface UseModelSessionResult {
  status: SessionStatus
  progress: FetchProgress | null
  handle: SessionHandle | null
  error: string | null
  /** Begin loading the model. Safe to call multiple times; subsequent calls during a
   *  load are ignored. */
  load: () => Promise<void>
  /** Run inference on the loaded session. */
  run: (feeds: FeedDict) => Promise<Record<string, TensorPayload> | null>
  /** Release the session on the worker. */
  dispose: () => Promise<void>
}

export function useModelSession(
  manifest: ModelManifest,
): UseModelSessionResult {
  const [status, setStatus] = useState<SessionStatus>('idle')
  const [progress, setProgress] = useState<FetchProgress | null>(null)
  const [handle, setHandle] = useState<SessionHandle | null>(null)
  const [error, setError] = useState<string | null>(null)
  const inFlight = useRef(false)
  const handleRef = useRef<SessionHandle | null>(null)
  handleRef.current = handle

  const load = useCallback(async () => {
    if (inFlight.current) return
    inFlight.current = true
    setError(null)
    setStatus('fetching')
    setProgress({ phase: 'check-cache', bytesLoaded: 0 })
    const onEvent = proxyCallback((event: FetchProgress | { phase: 'compiling' }) => {
      if (event.phase === 'compiling') {
        setStatus('compiling')
        return
      }
      setProgress(event)
      if (event.phase === 'ready') {
        // Stay in 'fetching' until compile starts; if the model was cached
        // we can jump straight to compiling on the next event.
      }
    })
    try {
      const h = await engine.loadSession(manifest, onEvent)
      setHandle(h)
      setStatus('ready')
    } catch (err) {
      setError((err as Error).message)
      setStatus('error')
    } finally {
      inFlight.current = false
    }
  }, [manifest])

  const run = useCallback<UseModelSessionResult['run']>(
    async (feeds) => {
      const h = handleRef.current
      if (!h) return null
      setStatus('running')
      try {
        const out = await engine.run(h.id, feeds)
        setStatus('ready')
        return out
      } catch (err) {
        setError((err as Error).message)
        setStatus('error')
        return null
      }
    },
    [],
  )

  const dispose = useCallback(async () => {
    const h = handleRef.current
    if (!h) return
    await engine.dispose(h.id)
    setHandle(null)
    setStatus('idle')
    setProgress(null)
  }, [])

  // Auto-dispose on unmount.
  useEffect(() => {
    return () => {
      const h = handleRef.current
      if (h) {
        void engine.dispose(h.id)
      }
    }
  }, [])

  return { status, progress, handle, error, load, run, dispose }
}

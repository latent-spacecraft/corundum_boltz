/**
 * GemLogo — a living word-mark vignette.
 *
 * Renders /public/gem.glb into a small WebGL canvas, slowly rotating, with
 * MeshPhysicalMaterial transmission for true refraction against a PMREM-baked
 * room environment. Source mesh is 234 verts; at 96 px the per-frame cost is
 * well under a millisecond and we share no resources with the Mol* canvas.
 *
 * Pauses when the tab is hidden (visibilitychange) and when the element
 * scrolls offscreen (IntersectionObserver). Recolored to ruby to match the
 * Corundum oxblood register; the source file's blue baseColor is overridden.
 */
import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js'
import { createGemMaterial, GEM_PRESETS } from './gemMaterial'

export interface GemLogoProps {
  size?: number
  onClick?: () => void
  title?: string
}

export function GemLogo({ size = 24, onClick, title = 'Corundum' }: GemLogoProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const dpr = Math.min(window.devicePixelRatio || 1, 2)

    const renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: true,
      premultipliedAlpha: true,
    })
    renderer.setPixelRatio(dpr)
    renderer.setSize(size, size, false)
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.15
    renderer.outputColorSpace = THREE.SRGBColorSpace

    const scene = new THREE.Scene()

    // Bake a small studio environment so transmission has something to refract.
    const pmrem = new THREE.PMREMGenerator(renderer)
    pmrem.compileEquirectangularShader()
    const envScene = new RoomEnvironment()
    const envRT = pmrem.fromScene(envScene, 0.04)
    scene.environment = envRT.texture

    const camera = new THREE.PerspectiveCamera(26, 1, 0.1, 100)
    // 3/4 view: lifted + offset so the facet topology reads on every rotation
    // frame instead of looking like a flat dart silhouette dead-on.
    camera.position.set(0, 0, 5.0)
    camera.lookAt(0, 0, 0)

    const root = new THREE.Group()
    scene.add(root)

    const ownedMaterials: THREE.Material[] = []
    let disposed = false

    const loader = new GLTFLoader()
    loader.load(
      '/asscher.glb',
      (gltf) => {
        if (disposed) return
        const gem = gltf.scene

        // Override the source material with a physical glass: ruby base
        // color, full transmission, low roughness, clearcoat. The original
        // file is sapphire-blue; we want oxblood register.
        gem.traverse((obj) => {
          const mesh = obj as THREE.Mesh
          if (mesh.isMesh) {
            // Threekit gem-material register, oxblood preset. Chromium-style
            // absorption (saturated red attenuationColor + short attenuation
            // distance) on a corundum-IOR base, with mild dispersion for the
            // spectral fringe you'd see on a real polished ruby. Shares the
            // same factory the structure gem-shell overlay will pull from.
            const mat = createGemMaterial(GEM_PRESETS.ruby)
            // Free the source material the loader created.
            const prev = mesh.material as THREE.Material | THREE.Material[]
            if (Array.isArray(prev)) prev.forEach((m) => m.dispose())
            else prev.dispose?.()
            mesh.material = mat
            ownedMaterials.push(mat)
          }
        })

        // Center & scale-fit so any future gem.glb edits still frame cleanly.
        const bbox = new THREE.Box3().setFromObject(gem)
        const center = bbox.getCenter(new THREE.Vector3())
        const dim = bbox.getSize(new THREE.Vector3())
        const maxDim = Math.max(dim.x, dim.y, dim.z) || 1
        gem.position.sub(center)
        // Visible square at z=0 spans ~2·tan(fov/2)·dist ≈ 1.48 units (fov 26°,
        // dist 3.2). Leave headroom for the rotation diagonal (×√2 ≈ 1.41) so
        // the gem never clips at any spin angle.
        const fit = 0.9 / maxDim
        gem.scale.multiplyScalar(fit)
        root.add(gem)
      },
      undefined,
      (err) => console.warn('[GemLogo] failed to load gem.glb', err),
    )

    let raf = 0
    let last = performance.now()
    let phase = 0
    let running = true

    const tick = (now: number) => {
      raf = requestAnimationFrame(tick)
      if (!running) {
        last = now
        return
      }
      const dt = Math.min(0.1, (now - last) / 1000)
      last = now
      phase += dt * -0.42 // rad/s ≈ 30 s/rotation
      //root.rotation.y = phase/4
      root.rotation.z = phase
      root.rotation.x = 0.22 // Math.sin(phase * 5.0) * 0.2
      renderer.render(scene, camera)
    }
    raf = requestAnimationFrame(tick)

    const onVis = () => {
      running = !document.hidden
      last = performance.now()
    }
    document.addEventListener('visibilitychange', onVis)

    const io = new IntersectionObserver(
      ([entry]) => {
        running = entry.isIntersecting && !document.hidden
        last = performance.now()
      },
      { threshold: 0.01 },
    )
    io.observe(canvas)

    return () => {
      disposed = true
      cancelAnimationFrame(raf)
      document.removeEventListener('visibilitychange', onVis)
      io.disconnect()
      ownedMaterials.forEach((m) => m.dispose())
      envRT.dispose?.()
      pmrem.dispose()
      renderer.dispose()
    }
  }, [size])

  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={`${title} — home`}
      title={title}
      style={{
        display: 'inline-block',
        padding: 0,
        margin: 0,
        background: 'transparent',
        border: 'none',
        cursor: onClick ? 'pointer' : 'default',
        lineHeight: 0,
        WebkitTapHighlightColor: 'transparent',
      }}
    >
      <canvas
        ref={canvasRef}
        style={{ width: size, height: size, display: 'block' }}
      />
    </button>
  )
}

# Corundum

> Browser-native AI inference for the life sciences. No install, no compute account, no data egress.

Corundum is a monorepo of standalone web apps that run AI models entirely in your browser tab. Weights download once, cache in OPFS, and execute on your own GPU via WebGPU (with a threaded-WASM fallback). The work you do never leaves the device.

## Packages

| Package | What it does |
| --- | --- |
| [`@corundum/boltz`](./packages/boltz) | AlphaFold3-class protein structure prediction (Boltz-2, MIT) running fully in-browser. |

## Quick start

```bash
git clone <this repo>
cd corundum
npm install
npm run dev
```

The dev script boots the Boltz app on `https://localhost:5173/` (self-signed cert via `@vitejs/plugin-basic-ssl` so WebGPU + SharedArrayBuffer are available on non-localhost too).

## License

MIT — see [LICENSE](./LICENSE). Each package may carry additional license notices for the upstream model weights it serves; see the package READMEs.

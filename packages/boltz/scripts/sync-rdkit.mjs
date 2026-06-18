#!/usr/bin/env node
//
// Copy the RDKit WASM artifacts from the installed @corundum/rdkit package into
// public/rdkit/, so they're served at a stable path (/rdkit/*) and survive the
// prod build's `rm -f dist/assets/*.wasm`. Run on dev/build (see package.json).
//
import { mkdirSync, copyFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const require = createRequire(import.meta.url)
const __dirname = path.dirname(fileURLToPath(import.meta.url))
const pkgDir = path.dirname(require.resolve('@corundum/rdkit/package.json'))
const outDir = path.resolve(__dirname, '..', 'public', 'rdkit')

mkdirSync(outDir, { recursive: true })
for (const f of ['RDKit_minimal.js', 'RDKit_minimal.wasm']) {
  copyFileSync(path.join(pkgDir, 'dist', f), path.join(outDir, f))
  console.log(`[sync-rdkit] ${f} -> public/rdkit/${f}`)
}

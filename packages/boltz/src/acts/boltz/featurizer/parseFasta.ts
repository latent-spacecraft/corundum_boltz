/**
 * Multiline FASTA → ChainInput[] parser with chain-type detection.
 *
 * Rules:
 *   - A line beginning with `>` opens a new chain. Header syntax:
 *       >name [type]
 *     where the first whitespace-token is the chain name and the second
 *     (optional) is one of `protein` / `rna` / `dna`. Any further tokens
 *     are kept as a free-form description.
 *   - With no type token, the chain type is inferred from the sequence
 *     alphabet via `detectType()` (RNA if contains U and ⊆ ACGU, DNA if
 *     ⊆ ACGT, else protein).
 *   - All non-comment, non-header lines until the next `>` are concatenated
 *     into the open chain's sequence (whitespace within a line is stripped).
 *   - Lines starting with `;` are FASTA comments — silently skipped.
 *   - Input with no `>` anywhere is treated as one unnamed chain whose
 *     type is detected from its alphabet (default protein for non-nucleic).
 *   - A `>` header followed by no sequence content is an error.
 *   - An unknown type token in a header is an error.
 *   - Empty / whitespace-only input is an error.
 */
import type { ChainInput } from './index'
import type { ChainType } from './tables'

const KNOWN_TYPES = new Set<ChainType>(['protein', 'rna', 'dna', 'ligand'])

/**
 * Lightweight alphabet heuristic for header-less or untyped chains.
 *   - U + only ACGU letters → RNA
 *   - Only ACGT letters     → DNA
 *   - Anything else         → protein
 *
 * Selenocysteine peptides (which contain U among other AA letters) fall
 * through to protein; ambiguous all-ACG strings default to DNA. Users with
 * either edge case should tag the header explicitly: `>name protein`.
 */
export function detectType(sequence: string): ChainType {
  const upper = sequence.toUpperCase()
  if (upper.length === 0) return 'protein'
  if (/U/.test(upper) && /^[ACGU]+$/.test(upper)) return 'rna'
  if (/^[ACGT]+$/.test(upper)) return 'dna'
  return 'protein'
}

export interface ParsedChain extends ChainInput {
  /** First token after `>` on the header line. Empty string for unnamed bare-sequence input. */
  name: string
  /** Full remainder of the header line after the name (and type, if present), trimmed. */
  description: string
  /** True if the chain type was set by an explicit header tag; false if detected from alphabet. */
  typeExplicit: boolean
}

export function parseFasta(text: string): ParsedChain[] {
  const lines = text.split(/\r?\n/)
  const chains: ParsedChain[] = []
  let current: ParsedChain | null = null

  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i]
    // FASTA comments — `;` at column 0. Strip and continue.
    if (raw.startsWith(';')) continue
    const trimmed = raw.trim()
    if (trimmed.length === 0) continue
    if (trimmed.startsWith('>')) {
      // Close out — we don't validate emptiness yet; lets a trailing header
      // with no body produce a precise error message at the end of the loop.
      current = parseHeader(trimmed)
      chains.push(current)
      continue
    }
    if (current === null) {
      // No header seen yet → start an implicit unnamed chain. We allow
      // subsequent `>` headers to open additional chains after this one,
      // though that's an unusual input shape. Type is filled in pass 2
      // by detectType() from the accumulated sequence.
      current = makeChain('', '', false)
      chains.push(current)
    }
    current.sequence += trimmed.replace(/\s+/g, '')
  }

  if (chains.length === 0) {
    throw new Error('FASTA input is empty')
  }
  for (const c of chains) {
    if (c.sequence.length === 0) {
      const label = c.name ? `'${c.name}'` : 'header'
      throw new Error(`FASTA chain ${label} has no sequence body`)
    }
  }
  // Pass 2: resolve type for chains that didn't have a header tag.
  for (const c of chains) {
    if (!c.typeExplicit) c.type = detectType(c.sequence)
  }
  return chains
}

function parseHeader(line: string): ParsedChain {
  // line starts with '>'; strip it.
  const after = line.slice(1).trimStart()
  if (after.length === 0) {
    return makeChain('', '', false)
  }
  const tokens = after.split(/\s+/)
  const name = tokens[0]
  // Look for an optional type tag in the second token.
  if (tokens.length >= 2 && KNOWN_TYPES.has(tokens[1].toLowerCase() as ChainType)) {
    const type = tokens[1].toLowerCase() as ChainType
    // `>name ligand smiles` marks the body as a SMILES string (preprocessed
    // via the endpoint) rather than a CCD code.
    if (type === 'ligand' && tokens[2]?.toLowerCase() === 'smiles') {
      const description = tokens.slice(3).join(' ').trim()
      const c = makeChain(name, description, true, type)
      c.ligandFormat = 'smiles'
      return c
    }
    const description = tokens.slice(2).join(' ').trim()
    return makeChain(name, description, true, type)
  }
  // Second token wasn't `protein`/`rna`/`dna` — treat the whole tail as
  // free-form description. The alphabet heuristic in `detectType()` will
  // classify the chain from its sequence content, so typos like
  // `>x prot` fall back gracefully rather than blocking the user.
  const description = tokens.slice(1).join(' ').trim()
  return makeChain(name, description, false)
}

function makeChain(
  name: string,
  description: string,
  typeExplicit: boolean,
  type: ChainType = 'protein',
): ParsedChain {
  return { name, description, sequence: '', type, typeExplicit }
}

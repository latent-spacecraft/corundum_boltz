/**
 * Forgiving FASTA parser.
 *
 * Accepts:
 *   - bare amino acid strings ("MKWVT...")
 *   - single-record FASTA (one ">header" line + sequence lines)
 *   - multi-record FASTA (returns all; the caller picks the first by default)
 *
 * Strips whitespace within sequence body, does not validate residue alphabet —
 * that's the tokenizer's job.
 */

export interface FastaRecord {
  /** Whatever followed the leading '>' on the header line, trimmed. Empty if no header. */
  header: string
  /** Residue string with no whitespace. */
  sequence: string
}

export function parseFasta(text: string): FastaRecord[] {
  const records: FastaRecord[] = []
  let current: FastaRecord | null = null
  const lines = text.split(/\r?\n/)
  for (const raw of lines) {
    const line = raw.trim()
    if (line.length === 0) continue
    if (line.startsWith('>')) {
      if (current) records.push(current)
      current = { header: line.slice(1).trim(), sequence: '' }
    } else {
      if (!current) current = { header: '', sequence: '' }
      current.sequence += line.replace(/\s+/g, '')
    }
  }
  if (current) records.push(current)
  return records
}

/** Convenience: take a chunk of user input and return the first sequence found. */
export function firstSequenceFrom(text: string): FastaRecord | null {
  const records = parseFasta(text)
  return records[0] ?? null
}

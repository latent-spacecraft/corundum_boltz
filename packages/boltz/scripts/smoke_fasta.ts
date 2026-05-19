/**
 * Smoke test for the FASTA parser.
 *
 * Covers: bare sequence (no header), single header + multi-line body,
 * multiple chains, comment (`;`) lines, blank lines, whitespace inside
 * sequence lines, empty body after header (should throw), empty input
 * (should throw), header with only a `>` (anonymous chain), header with
 * name + description.
 *
 * Run: `npm run smoke:fasta` from `packages/boltz`.
 */
import { parseFasta, detectType } from '../src/acts/boltz/featurizer/parseFasta'

function expect(name: string, ok: boolean, detail?: string) {
  const flag = ok ? '✓' : '✗'
  console.log(`  ${flag} ${name}${detail ? ' — ' + detail : ''}`)
  if (!ok) process.exitCode = 1
}

function expectThrow(name: string, fn: () => unknown, msgIncludes?: string) {
  try {
    fn()
    expect(name, false, 'expected throw, got success')
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e)
    if (msgIncludes && !msg.includes(msgIncludes)) {
      expect(name, false, `error message lacks '${msgIncludes}': ${msg}`)
    } else {
      expect(name, true, `(${msg})`)
    }
  }
}

console.log('— Bare sequence (no header) —')
{
  const chains = parseFasta('MKWVTFISLLFLFSSAYS')
  expect('one chain', chains.length === 1)
  expect('name is empty', chains[0].name === '')
  expect('sequence intact', chains[0].sequence === 'MKWVTFISLLFLFSSAYS')
  expect('type protein', chains[0].type === 'protein')
}

console.log('— Single chain, header + multi-line body —')
{
  const text = '>chain_a\nMKWVTFISL\nLFLFSSAYS'
  const chains = parseFasta(text)
  expect('one chain', chains.length === 1)
  expect("name is 'chain_a'", chains[0].name === 'chain_a')
  expect('description empty', chains[0].description === '')
  expect('body concatenated', chains[0].sequence === 'MKWVTFISLLFLFSSAYS')
}

console.log('— Header with name + description —')
{
  const chains = parseFasta('>chainA   first chain | source=test\nMKW')
  expect("name is 'chainA'", chains[0].name === 'chainA')
  expect('description preserved', chains[0].description === 'first chain | source=test')
}

console.log('— Multiple chains —')
{
  const text = [
    '>chain_a heavy chain',
    'NLYIQWLKDGGPSSGR',
    'PPPS',
    '',
    '; this is a comment line',
    '>chain_b',
    'MKWVTFISLLFLFSSAYS',
  ].join('\n')
  const chains = parseFasta(text)
  expect('two chains', chains.length === 2)
  expect('chain a sequence', chains[0].sequence === 'NLYIQWLKDGGPSSGRPPPS')
  expect('chain a description', chains[0].description === 'heavy chain')
  expect('chain b sequence', chains[1].sequence === 'MKWVTFISLLFLFSSAYS')
  expect('chain b description empty', chains[1].description === '')
}

console.log('— Whitespace inside body lines is stripped —')
{
  const chains = parseFasta('>x\nMK W V TF\n  ISLLF  LFSSAYS')
  expect('stripped', chains[0].sequence === 'MKWVTFISLLFLFSSAYS')
}

console.log('— Header with only `>` (anonymous) —')
{
  const chains = parseFasta('>\nMKW')
  expect('name is empty', chains[0].name === '')
  expect('description empty', chains[0].description === '')
  expect('sequence ok', chains[0].sequence === 'MKW')
}

console.log('— Type detection heuristic —')
{
  expect("'MKWVTF' → protein", detectType('MKWVTF') === 'protein')
  expect("'GGGAACCC' (no U) → DNA (only ACGT)", detectType('GGGAACCC') === 'dna')
  expect("'GGGAAUCC' (has U, only ACGU) → RNA", detectType('GGGAAUCC') === 'rna')
  expect("'ACGTACGT' → DNA", detectType('ACGTACGT') === 'dna')
  expect("'MUCKLPS' (U in protein) → protein (mixed alphabet)",
    detectType('MUCKLPS') === 'protein')
  expect("'ACG' → DNA (in ACGT alphabet)", detectType('ACG') === 'dna')
  expect("empty → protein", detectType('') === 'protein')
}

console.log('— Explicit type tag in header —')
{
  const chains = parseFasta([
    '>myrna rna',
    'GGGAAUCC',
    '>mydna dna',
    'AAAATTTT',
    '>myprot protein',
    'AGCT', // alphabet would say DNA but tag says protein
  ].join('\n'))
  expect('three chains', chains.length === 3)
  expect('rna chain type rna', chains[0].type === 'rna')
  expect('rna chain typeExplicit', chains[0].typeExplicit === true)
  expect('dna chain type dna', chains[1].type === 'dna')
  expect('protein tag overrides alphabet', chains[2].type === 'protein')
  expect('protein chain typeExplicit', chains[2].typeExplicit === true)
}

console.log('— Header description coexists with type tag —')
{
  const chains = parseFasta('>chain_a protein heavy chain serotype 3\nMKW')
  expect('type protein', chains[0].type === 'protein')
  expect('typeExplicit', chains[0].typeExplicit === true)
  expect('description preserved', chains[0].description === 'heavy chain serotype 3')
}

console.log('— Unrecognised second token is description, not error —')
{
  const chains = parseFasta('>x prot\nMKW')
  expect('type detected from alphabet', chains[0].type === 'protein')
  expect('typeExplicit false (heuristic)', chains[0].typeExplicit === false)
  expect("'prot' kept as description", chains[0].description === 'prot')
}

console.log('— Auto-detection on header-less input —')
{
  expect('bare protein → protein',
    parseFasta('MKWVTF')[0].type === 'protein')
  expect('bare RNA → rna',
    parseFasta('GGGAAUCC')[0].type === 'rna')
  expect('bare DNA → dna',
    parseFasta('AAAATTTT')[0].type === 'dna')
}

console.log('— Errors —')
{
  expectThrow('empty input', () => parseFasta(''), 'empty')
  expectThrow('whitespace-only input', () => parseFasta('   \n  \n'), 'empty')
  expectThrow(
    'header with no body',
    () => parseFasta('>only_header'),
    'no sequence body',
  )
  expectThrow(
    'second chain has no body',
    () => parseFasta('>a\nMKW\n>b\n'),
    'no sequence body',
  )
}

console.log(process.exitCode ? '\nFAIL' : '\nOK')

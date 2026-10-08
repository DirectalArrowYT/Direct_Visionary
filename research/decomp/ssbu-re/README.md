# SSBU reverse-engineering tools

This directory contains reusable scripts for inspecting SSBU executable data with
Ghidra.

Set `SSBU_DUMP_DIR` to the directory containing the input files:

```bash
export SSBU_DUMP_DIR=/path/to/external/ssbu-dumps
```

On Windows PowerShell, use:

```powershell
$env:SSBU_DUMP_DIR = "C:\path\to\external\ssbu-dumps"
```

Depending on the script, the expected files are `exefs/main`,
`main_decompressed.bin`, and `main_reloc.bin`. Run `nso_image.py` to generate
`main_decompressed.bin` in the selected input directory.

Use `ghidra_proj/` for the local Ghidra project. Analysis state and generated
output remain local to this working directory.

Run the scripts in `gscripts/` with this directory selected as Ghidra's working
directory. Generated text output is written relative to this directory.

## Decompiling the particle library

SSBU links NintendoWare Vfx 8.2.1 (`nn::vfx`) statically; in 13.0.3 it sits at roughly
`0x86000..0xafc00` of `.text`, between NintendoWare Font and G3d. To dump it:

1. `python nso_image.py` then `python nso_reloc.py` — writes `main_reloc.bin`, the image with
   its relative relocations applied so vtables and pointer tables are populated.
2. Import `main_reloc.bin` into `ghidra_proj/` as raw `AARCH64:LE:64:v8A` at base 0, without
   analysis.
3. `python vfx_targets.py [start end]` — writes `vfx_targets.txt`.
4. Run `gscripts/DumpVfx.py` on the imported program — writes `vfx_decomp.txt`.

Addresses in the older scripts here were measured on other builds (13.0.4 among them) and do
not line up with 13.0.3.

`vfx_asm.py <start> [end]` disassembles a range with emitter field names attached, which reads
better than the decompiler for the library's NEON float code. The names come from
`emitter_offsets_v22.txt`, produced by
`cargo run --release --example emitter_offsets -- <eff>` in `vendor/effect_library`; the
runtime's resource pointer is the start of the EMTR section's binary data, so those offsets
are the ones the code uses.

### What has been read (13.0.3)

| Address | What it is |
| --- | --- |
| `0x8dc80` | Emitter update: emission window, one-time rule, per-frame bookkeeping |
| `0x8d340` | Emission: gap check, rate less a random percentage, fraction carried over |
| `0x88750` | Next gap: `interval + 1 +` a random whole number below `interval_random` |
| `0x90bf0` | One firing: evenly divided shapes multiply the count by their divisions |
| `0x8fbe0` | One particle: position, velocity, scale, life and their random percentages |
| `0x4f48c30` | Table of the 16 emitter shape functions, `0x90c00` (point) to `0x93070` (primitive) |
| `0x93220` | Sweep angle helper shared by the sphere shapes |
| `0x95370` | Per-frame particle step: move, drag, gravity |
| `0x990a0` | Builds the two 512-entry random vector tables (xorshift, fixed seeds) |
| `0x8b480` | Copies rotation and other emitter fields into the shader's uniform block (not read yet) |

The emitter's random numbers are `x = x * 0x41C64E6D + 12345` on a per-emitter state at
`emitter + 0xac`, used as `state * 2^-32` before advancing. `src/eff_sim.rs` implements the
rules read from the first eight rows.

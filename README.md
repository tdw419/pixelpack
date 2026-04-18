# pixelpack

One pixel. A working program.

A single pixel (32-bit RGBA) encodes a generative recipe that, when booted, expands into a runnable program. This is not compression. The pixel stores a formula, not data. Like DNA, the pixel is a seed that needs cellular machinery to become something.

## The Idea

A pixel has 4 channels: R, G, B, A. That's 32 bits. 32 bits is not enough to store a program. But 32 bits IS enough to store a recipe -- a set of instructions for a generative process that produces a program.

The recipe works because of **shared context**. The pixel alone is meaningless, just like DNA without a cell. The "cellular machinery" here is a set of 16 expansion strategies -- recipes for generating bytes from a small number of parameters. The pixel selects which recipe to run and supplies the parameters.

```
32-bit seed (one pixel)
    |
    +-- top 4 bits: which recipe to use (0-F)
    +-- bottom 28 bits: parameters for that recipe
    |
    v
Recipe executes
    |
    v
Bytes come out
    |
    v
Those bytes ARE a working program
```

## Phase 2: Multi-Pixel Encoding

Phase 1 proved the mechanism: one pixel = one program (3-15 bytes). Phase 2 scales it up:

**Multi-pixel images** encode programs from 3 to 200+ bytes. A 2x2 image gives 4 seeds (128 bits), a 4x4 gives 16 seeds (512 bits). Each pixel expands independently and results concatenate.

**Auto-growing dictionary** builds entries from a corpus of real programs. V2 dictionary has 96 entries (V1 had 32). Entries 0-31 are frozen for backward compatibility.

**Smart segmentation** splits a target program into segments, finds seeds for each, and packs them into a multi-pixel PNG.

```
Target program (34 bytes)
    |
    v
Segmenter splits into N pieces
    |
    v
Each piece -> 32-bit seed via strategies
    |
    v
Seeds -> NxM PNG pixels
    |
    v
Decoding: read PNG, extract seeds, expand each, concatenate
```

## How It Works

### The Seed Format

Every 32-bit seed is split into two parts:

```
[SSSS] [PPPPPPPPPPPPPPPPPPPPPPPPPPPP]
 ^---^ ^----------------------------^
 4 bits        28 bits
 strategy      parameters
```

- **Strategy** (top 4 bits, 0x0-0xF): selects one of 16 expansion methods
- **Parameters** (bottom 28 bits): feeds into the chosen strategy

### The 16 Strategies

The strategies are the "genome" -- shared context that gives meaning to the seed.

| Hex | Name | What it does | Max output |
|-----|------|-------------|------------|
| 0-6 | DICT_N | Concatenate N entries (1-7) from a 16-word dictionary using 4-bit indices | ~35 bytes |
| 7 | NIBBLE | 7 nibbles each look up a byte in a common-symbol table | 7 bytes |
| 8 | DICTX5 | 5 entries from a 32-word extended dictionary using 5-bit indices | ~40 bytes |
| 9 | DICTX6 | 6 entries from extended dictionary entries 16-31 | ~18 bytes |
| A | DICTX7 | 7 entries from extended dictionary entries 16-31 | ~21 bytes |
| B | RLE | Run-length encoded patterns (byte A x count A + byte B x count B, repeated) | ~512 bytes |
| C | XOR_CHAIN | XOR chain: start byte, each next = (prev XOR key) AND mask | 16 bytes |
| D | LINEAR | Linear sequence: start, start+step, start+2*step, with XOR modifier | 16 bytes |
| E | BYTEPACK | Direct byte encoding with 8 sub-modes (raw, XOR delta, ADD delta, nibble, etc.) | 7 bytes |
| F | TEMPLATE | XOR substitution on 16 built-in program templates + 2 extra bytes | ~16 bytes |

### Multi-Pixel Encoding

Phase 2 adds multi-pixel support via two new files:

- `expand2.py` -- `expand_multi()` chains multiple seeds into one output, `expand_from_png()` decodes any PNG
- `boot2.py` -- `make_multipixel_png()` creates NxM RGBA PNGs, `encode_multi()` auto-segments and encodes. CLI: `encode`, `decode`, `demo`
- `verify2.py` -- 20 tests covering V1 backward compat (6) + V2 multi-pixel (14)

The image dimensions are chosen automatically:
- 1 seed -> 1x1 (same as V1)
- 2 seeds -> 2x1
- 3-4 seeds -> 2x2
- 5-9 seeds -> 3x3
- 10-16 seeds -> 4x4

### The Dictionary

The core strategies (0x0-0x6) use a shared 16-entry dictionary of programming fragments:

```
Index  Fragment   Length
0      print(       6
1      )            1
2      "            1
3      Hello        5
4      \n           1
5      echo         5
6      World        5
7      def          4
8      42           2
9      main         4
10     ()           2
11     ,            2
12     !            1
13     void         5
14     {            1
15     }            1
```

An extended dictionary adds 16 more entries (x, =, +, -, *, ;, 1, 0, if, return, int, for, while, class, space, fn) for strategies 8-A.

## The Pipeline

```
ENCODE (file -> multi-pixel PNG):
  1. Read target bytes
  2. Try single seed first (V1 path)
  3. If no single seed, segment the target
  4. For each segment, find a seed via strategy inversion
  5. Pack all seeds into NxM RGBA PNG
  6. PNG is a viewable image!

DECODE (multi-pixel PNG -> file):
  1. Read PNG dimensions and pixel data
  2. Extract RGBA from each pixel -> 32-bit seeds
  3. Expand each seed via its strategy
  4. Concatenate all expansions
  5. Output bytes are the original program
```

## Proven Examples

### Phase 1 (single pixel, 6/6 pass)

| Target program | Pixels | Seed | Size |
|---------------|--------|------|------|
| `print("Hello")\n` | 1x1 | 0x50412320 | 15B from 32 bits |
| `echo Hello\n` | 1x1 | 0x20000435 | 11B from 32 bits |
| `42\n` | 1x1 | 0x10000048 | 3B from 32 bits |
| `Hello, World!\n` | 1x1 | 0x4004C6B3 | 14B from 32 bits |
| `print(42)\n` | 1x1 | 0x30004180 | 10B from 32 bits |
| `void main(){}\n` | 1x1 | 0x504FEA9D | 14B from 32 bits |

### Phase 2 (multi-pixel, 14/14 pass)

| Target program | Pixels | Seeds | Size |
|---------------|--------|-------|------|
| `x = "Hello"\nprint(x)\n` | 2x2 | 3 | 21B from 96 bits |
| `print("Hello")\necho Hello\n` | 2x1 | 2 | 26B from 64 bits |
| `print("Hello")\nprint(42)\n` | 2x1 | 2 | 25B from 64 bits |
| `int main(){puts("Hello");}\n` | 3x3 | 8 | 27B from 256 bits |
| `PSET 10 20\nCOLOR 255 0 0\nDRAW\n` | 3x3 | 8 | 30B from 256 bits |
| `def greet(name):\n    print(name)\n\n` | 4x4 | 10 | 34B from 320 bits |
| `for i in range(10):\n    print(i)\n\n` | 4x4 | 10 | 34B from 320 bits |
| `def add(a, b):\n    return a + b\n\nprint(add(1, 2))\n` | 4x4 | 12 | 50B from 384 bits |
| `x = 1\ny = 2\nif x > 0:\n    print(y)\n` | 3x3 | 7 | 35B from 224 bits |
| `def greet(name):\n    print("Hello, " + name)\n\ngreet("World")\n` | 4x4 | 12 | 61B from 384 bits |
| `#!/bin/bash\nfor i in 1 2 3; do\n  echo "Number: $i"\ndone\n` | 4x4 | 13 | 73B from 416 bits |
| `#include <stdio.h>\nint main(){...}\n` (C program) | 5x5 | 20 | 103B from 640 bits |
| Python class with methods (175B) | 7x7 | 41 | 175B from 2624 bits |
| Python fibonacci (254B) | 8x8 | 56 | 254B from 3584 bits |

All Python programs run and produce correct output after full encode/decode PNG round-trip. Total: **20/20 tests passing**.

## The Key Insight

This is NOT compression. Compression takes N bits and makes them smaller. This takes 32 bits and GENERATES more bits from nothing but a recipe.

The reason it works: most of the "meaning" lives in the shared context (the strategies + dictionary), not in the seed. The seed is a pointer into a space of known recipes. Like how "JMP 0x42" means nothing without a CPU to execute it, the pixel means nothing without the expander.

This is the DNA analogy made literal:
- DNA (the pixel): a small sequence that encodes instructions
- Ribosomes (the expander): machinery that reads those instructions and produces proteins (bytes)
- The genome (dictionary + strategies): shared infrastructure that makes the instructions meaningful

## Files

```
pixelpack/
  expand.py       The SEED-VM. 16 expansion strategies. Takes a 32-bit seed, produces bytes.
  find_seed.py    Analytical seed search. Inverts each strategy to find seeds for target bytes.
  boot.py         Single-pixel PNG encoder/decoder. Writes a 1x1 RGBA PNG, reads it back.
  verify.py       Phase 1 round-trip tests. 6 targets, all pass.

  expand2.py      Multi-pixel expansion. Chains multiple seeds, reads multi-pixel PNGs.
  boot2.py        Multi-pixel PNG encoder/decoder. Auto-segments, creates NxM PNGs. CLI: encode/decode/demo.
  verify2.py      Phase 2 round-trip tests. 20 targets (6 V1 + 14 V2), all pass.
```

## Usage

```bash
# Phase 1: Verify the proof
python3 verify.py

# Phase 2: Verify multi-pixel encoding
python3 verify2.py

# Encode a program into pixels (tries single, then multi)
python3 boot2.py encode program.py output.png

# Decode pixels back into a program
python3 boot2.py decode output.png recovered.py

# Run the phase 2 demo
python3 boot2.py demo

# Expand a seed directly
python3 expand.py 50412320
```

## Limitations (Honest Assessment)

1. **Multi-pixel efficiency varies.** Programs that decompose into dictionary fragments need few seeds. Programs with lots of non-dictionary characters need many (one BYTEPACK per 3-5 bytes). A 34-byte program with rare characters needs ~10 seeds.

2. **The dictionary is fixed.** The 16+16 entry dictionaries are hand-picked programming fragments. They work well for Python/C/shell but poorly for arbitrary text. Domain-specific dictionaries would improve efficiency.

3. **Not a general-purpose encoder.** You can encode any bytes, but the pixel count scales linearly with non-dictionary content. This is a generative system, not a compressor.

## History

Phase 1 was built with the Recursive Feedback Loop (RFL) -- 5 iterations over 71 minutes. Phase 2 adds multi-pixel support and scales from 15-byte single-pixel programs to 254-byte multi-pixel programs with full PNG round-trip verification.

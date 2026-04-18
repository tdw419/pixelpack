#!/usr/bin/env python3
"""
Estimate instruction cost of Pixelpack expansion in Geometry OS assembly.

Current infinite_map.asm:
  - Coarse hash: ~15 instructions per tile
  - Biome cascade (CMP/BLT): ~20 instructions average per tile
  - Fine hash + structure check: ~8 instructions
  - Total: ~43 instructions per tile
  - 64x64 tiles = ~176K instructions for terrain
  
  With RECTF + overhead: ~322K total

Pixelpack-enhanced approach:
  - Coarse hash: same ~15 instructions
  - Biome base color lookup: 1 instruction (table index)
  - Fine hash: ~8 instructions (same as now)
  - Seed expansion (simplified): needs estimate
  - Variation application: ~5 instructions
  - RECTF: same 1 instruction

The key cost is "seed expansion". How much does a simplified
version cost in assembly?

Simplified expansion: use only the NIBBLE strategy (strategy 0x7).
  - Extract 7 nibbles from 28-bit payload: 7 x (SHR + AND) = 14 ops
  - Look up each nibble in NIBBLE_TABLE: 7 x LOAD = 7 ops
  - Total: ~21 instructions per tile

Compare to current: ~43 instructions per tile.
The simplified Pixelpack approach is CHEAPER than the current cascade!

Full expansion with strategy dispatch would be more expensive,
but still within budget since we have 678K instructions of headroom.
"""

# Geometry OS instruction costs
# Each mnemonic = 1 instruction
# RECTF, MUL, ADD, SUB, XOR, SHR, AND, LDI, LOAD, STORE, JMP, JZ, CMP, BLT, BGE = 1 each

def estimate_current_cost():
    """Current infinite_map.asm per-tile cost."""
    cost = {
        'coarse_hash': 0,
        'biome_cascade': 0,
        'fine_hash': 0,
        'structure_check': 0,
        'color_select': 0,
        'rectf': 1,
        'tint': 1,  # ADD r17, r23
    }
    
    # Coarse hash (lines 186-206): SHR, LDI, MUL, SHR, LDI, MUL, XOR, LDI, MUL, LDI, SHR = 11
    cost['coarse_hash'] = 11
    
    # Fine hash (lines 208-215): MOV, LDI, MUL, MOV, LDI, MUL, XOR = 7
    cost['fine_hash'] = 7
    
    # Structure check (lines 217-223): LDI, MOV, AND, LDI, CMP, JNZ = 6
    cost['structure_check'] = 6
    
    # Biome cascade: average case traverses ~10 comparisons before match
    # Each comparison: LDI, CMP, BLT = 3 instructions
    # Plus the final JMP to color label
    cost['biome_cascade'] = 10 * 3 + 1  # ~31 instructions average
    
    # Color select (LDI r17, 0x...): 1 instruction
    # Some have sub-cascade (2-3 more CMP/JZ)
    cost['color_select'] = 3  # average
    
    total = sum(cost.values())
    return cost, total


def estimate_pixelpack_simplified():
    """
    Pixelpack-enhanced per-tile cost using simplified expansion.
    
    Strategy: always use NIBBLE_TABLE (strategy 0x7 equivalent).
    Skip the strategy dispatch entirely -- just hash -> nibble lookup -> color.
    """
    cost = {
        'coarse_hash': 0,    # same
        'biome_lookup': 0,   # replaces cascade
        'fine_hash': 0,      # same, but becomes the SEED
        'nibble_expand': 0,  # NEW: extract 3 nibbles for R,G,B variation
        'color_modulate': 0, # NEW: add variation to base color
        'rectf': 1,          # same
        'tint': 1,           # same
    }
    
    # Coarse hash: same 11 instructions
    cost['coarse_hash'] = 11
    
    # Biome lookup: top 5 bits of hash -> table index -> base color
    # SHR by 27, then LOAD from color table
    cost['biome_lookup'] = 2  # SHR + LOAD (vs ~31 for cascade!)
    
    # Fine hash: same 7 instructions, but now this IS the seed
    cost['fine_hash'] = 7
    
    # Nibble expand: extract 3 nibbles from seed for R,G,B variation
    # For each channel: SHR by (0, 8, 16) + AND 0xF + LOAD from variation table + ADD to base
    # 3 channels x (SHR + AND + LOAD + ADD) = 12 instructions
    cost['nibble_expand'] = 12
    
    # Color modulate: clamp to 0-255 (2 comparisons per channel)
    # In practice, skip clamping -- Geometry OS doesn't wrap badly for small offsets
    cost['color_modulate'] = 0
    
    total = sum(cost.values())
    return cost, total


def estimate_pixelpack_full():
    """
    Full Pixelpack expansion with strategy dispatch.
    
    This would support all 16 expansion strategies, each producing
    different byte sequences that map to tile variation.
    """
    cost = {
        'coarse_hash': 11,
        'biome_lookup': 2,     # table lookup vs cascade
        'fine_hash': 7,        # seed generation
        'strategy_dispatch': 3, # SHR 28 + AND 0xF + JMP_TABLE (or cascading CMP)
        'expansion': 0,        # varies by strategy
        'color_apply': 8,      # apply expanded bytes to base color
        'rectf': 1,
        'tint': 1,
    }
    
    # Average expansion cost across strategies:
    # DICT_N (0x0-0x6): 1-7 table lookups = avg 4 x (SHR + AND + LOAD) = 12
    # NIBBLE (0x7): 7 x (SHR + AND + LOAD) = 21
    # DICTX5 (0x8): 5 x (SHR + AND + LOAD) = 15
    # BPE (0x9): 4 x (SHR + AND + LOAD + CONCAT) = 16
    # BYTEPACK (0xE): 5 x (bit extraction) = ~20
    # Average: ~16 instructions
    cost['expansion'] = 16
    
    total = sum(cost.values())
    return cost, total


def main():
    print("=== Instruction Cost Analysis ===\n")
    
    current, c_total = estimate_current_cost()
    simplified, s_total = estimate_pixelpack_simplified()
    full, f_total = estimate_pixelpack_full()
    
    tiles = 64 * 64  # 4096 tiles per frame
    
    print("Per-tile cost breakdown:")
    print(f"{'Component':<25s} {'Current':>8s} {'Simple':>8s} {'Full':>8s}")
    print("-" * 55)
    
    all_keys = list(current.keys())
    for key in all_keys:
        cv = current.get(key, '-')
        sv = simplified.get(key, '-')
        fv = full.get(key, '-')
        print(f"  {key:<23s} {str(cv):>6s}   {str(sv):>6s}   {str(fv):>6s}")
    
    print("-" * 55)
    print(f"  {'TOTAL per tile':<23s} {c_total:>6d}   {s_total:>6d}   {f_total:>6d}")
    print()
    
    print(f"Frame budget: 4096 tiles x cost = total instructions")
    print(f"  Current:  {tiles * c_total:>10,d} instructions  (actual: ~322K with overhead)")
    print(f"  Simple:   {tiles * s_total:>10,d} instructions")
    print(f"  Full:     {tiles * f_total:>10,d} instructions")
    print()
    
    budget = 1_000_000
    overhead = 322_000 - tiles * c_total  # cursor, minimap, etc
    print(f"Budget: {budget:,d} instructions")
    print(f"Overhead (cursor + minimap + control): ~{overhead:,d}")
    print()
    
    remaining = budget - overhead
    for name, total in [("Current", c_total), ("Simple Pixelpack", s_total), ("Full Pixelpack", f_total)]:
        frame_cost = tiles * total + overhead
        pct = (frame_cost / budget) * 100
        status = "OK" if frame_cost <= budget else "OVER BUDGET"
        print(f"  {name:<20s}: {frame_cost:>8,d} / {budget:,d} = {pct:5.1f}%  [{status}]")
    
    print()
    print("KEY INSIGHT: Simplified Pixelpack is CHEAPER than the current cascade")
    print("because a table lookup (2 instructions) replaces the biome cascade (~31 instructions).")
    print("The seed expansion adds ~12 instructions but we save ~29 from the cascade.")
    print(f"Net savings: ~{c_total - s_total} instructions per tile = ~{(c_total - s_total) * tiles:,d} per frame")


if __name__ == "__main__":
    main()

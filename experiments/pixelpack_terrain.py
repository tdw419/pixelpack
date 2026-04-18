#!/usr/bin/env python3
"""
pixelpack_terrain.py -- Pixelpack seed-driven terrain tile generation

The insight: infinite_map.asm already computes a full 32-bit hash per tile,
then throws away 27 bits (only uses top 5 for biome index). Pixelpack
treats that same 32-bit hash as a SEED and expands it to 3-8 bytes of
rich tile detail -- base color, variation, pattern, structure flag.

This prototype shows the difference:
  - Original: 1 color per 8x8 biome zone (32x32 pixels)
  - Enhanced: Pixelpack expansion creates per-tile variation WITHIN biomes
"""

from expand import (
    seed_to_rgba, seed_from_rgba, expand,
    DICTIONARY, DICTIONARY_EXT, SUB_DICT, NIBBLE_TABLE,
    BPE_PAIR_TABLE
)

MASK = 0xFFFFFFFF

# 21 biome base colors from infinite_map.asm
BIOME_COLORS = [
    (0x00, 0x00, 0x44),  # 0  deep ocean
    (0x00, 0x00, 0xBB),  # 1  shallow water
    (0xC2, 0xB2, 0x80),  # 2  beach
    (0xDD, 0xBB, 0x44),  # 3  desert
    (0xCC, 0xAA, 0x33),  # 4  dunes
    (0x22, 0xAA, 0x55),  # 5  oasis
    (0x55, 0xBB, 0x33),  # 6  light grass
    (0x22, 0x88, 0x11),  # 7  dark grass
    (0x44, 0x55, 0x22),  # 8  swamp
    (0x2D, 0x4A, 0x1A),  # 9  mangrove
    (0x11, 0x66, 0x00),  # 10 forest
    (0x0A, 0x44, 0x00),  # 11 dense forest
    (0x88, 0x33, 0x88),  # 12 mushroom
    (0x66, 0x77, 0x66),  # 13 foothills
    (0x99, 0x99, 0x99),  # 14 mountain
    (0x88, 0x99, 0xAA),  # 15 tundra
    (0xFF, 0x33, 0x00),  # 16 lava
    (0x33, 0x22, 0x22),  # 17 basalt
    (0x44, 0x22, 0x11),  # 18 volcanic
    (0xCC, 0xCC, 0xEE),  # 19 snow
    (0xDD, 0xEE, 0xFF),  # 20 ice
    (0xFF, 0xFF, 0xFF),  # 21 peak
    (0x33, 0x77, 0xAA),  # 22 coral
    (0x77, 0x66, 0x55),  # 23 ruins
    (0x1A, 0x33, 0x33),  # 24 cavern
    (0x2A, 0x55, 0x55),  # 25 crystal cave
    (0x44, 0x44, 0x44),  # 26 ash
    (0x3D, 0x2B, 0x1F),  # 27 deadlands
    (0x4A, 0x35, 0x25),  # 28 barren
    (0x00, 0x44, 0x33),  # 29 fungal
    (0x00, 0x66, 0x55),  # 30 glowing
    (0x11, 0x00, 0x22),  # 31 void
]

BIOME_NAMES = [
    "deep_ocean", "shallow_water", "beach", "desert", "dunes", "oasis",
    "light_grass", "dark_grass", "swamp", "mangrove", "forest", "dense_forest",
    "mushroom", "foothills", "mountain", "tundra", "lava", "basalt",
    "volcanic", "snow", "ice", "peak", "coral", "ruins", "cavern",
    "crystal_cave", "ash", "deadlands", "barren", "fungal", "glowing", "void"
]


def coarse_hash(world_x, world_y):
    """Same hash as infinite_map.asm: coarse 8x8 zone -> biome index 0-31."""
    cx = world_x >> 3
    cy = world_y >> 3
    ch = ((cx * 99001) & MASK) ^ ((cy * 79007) & MASK)
    h = ((ch * 1103515245) & MASK)
    return h


def fine_hash(world_x, world_y):
    """Fine hash for per-tile detail (different primes)."""
    h1 = ((world_x * 374761393) & MASK)
    h2 = ((world_y * 668265263) & MASK)
    return (h1 ^ h2) & MASK


def seed_tile_color(world_x, world_y, frame=0):
    """
    Compute tile color using Pixelpack-style seed expansion.
    
    1. Coarse hash -> biome base color (same as infinite_map)
    2. Fine hash -> seed -> expand -> per-tile variation bytes
    3. Combine: base color + seed-derived variation
    
    This is what changes: instead of the same color for every tile
    in an 8x8 biome zone, each tile gets unique variation derived
    from the Pixelpack expansion of its coordinate seed.
    """
    # Biome from coarse hash (same as infinite_map.asm)
    ch = coarse_hash(world_x, world_y)
    biome = ch >> 27  # top 5 bits = biome index
    base_r, base_g, base_b = BIOME_COLORS[biome]
    
    # Per-tile seed from fine hash ( Pixelpack expansion )
    seed = fine_hash(world_x, world_y)
    
    # Expand the seed -- this is the Pixelpack integration
    expanded = expand(seed)
    
    if len(expanded) < 2:
        return (base_r, base_g, base_b)
    
    # Use expansion bytes as color modulation
    # Byte 0: R variation, Byte 1: G variation, Byte 2: B variation
    # Map expansion bytes (0-255) to small offsets (-16 to +15)
    mod_r = ((expanded[0] & 0x1F) - 16) if len(expanded) > 0 else 0
    mod_g = ((expanded[1] & 0x1F) - 16) if len(expanded) > 1 else 0
    mod_b = ((expanded[2 % len(expanded)] & 0x1F) - 16) if len(expanded) > 2 else 0
    
    # Apply structure flag (if byte 3 has high bit set, stronger variation)
    if len(expanded) > 3 and (expanded[3] & 0x80):
        mod_r *= 2
        mod_g *= 2
        mod_b *= 2
    
    r = max(0, min(255, base_r + mod_r))
    g = max(0, min(255, base_g + mod_g))
    b = max(0, min(255, base_b + mod_b))
    
    return (r, g, b)


def render_ascii(width=64, height=32, cam_x=0, cam_y=0, use_variation=True):
    """Render terrain as ASCII art. use_variation=False = original flat look."""
    for ty in range(height):
        row = ""
        for tx in range(width):
            wx = cam_x + tx
            wy = cam_y + ty
            
            if use_variation:
                r, g, b = seed_tile_color(wx, wy)
            else:
                # Original: flat biome color, no variation
                ch = coarse_hash(wx, wy)
                biome = ch >> 27
                r, g, b = BIOME_COLORS[biome]
            
            # ASCII brightness
            brightness = (r + g + b) / 3
            if brightness < 20:
                row += " "
            elif brightness < 50:
                row += "░"
            elif brightness < 80:
                row += "▒"
            elif brightness < 120:
                row += "▓"
            elif brightness < 180:
                row += "█"
            else:
                row += "▓"
        print(row)


def demo():
    print("=== Pixelpack Terrain: Flat vs Seed-Expanded ===\n")
    
    print("ORIGINAL (flat biome colors, no per-tile variation):")
    print("Each 8x8 zone is a single color\n")
    render_ascii(64, 16, cam_x=100, cam_y=200, use_variation=False)
    
    print("\n\nPIXELPACK ENHANCED (seed expansion drives per-tile variation):")
    print("Same biomes, but Pixelpack seeds create visual texture\n")
    render_ascii(64, 16, cam_x=100, cam_y=200, use_variation=True)
    
    print("\n\n--- Seed detail at specific tiles ---")
    for wx, wy in [(100, 200), (108, 200), (115, 207), (120, 215)]:
        ch = coarse_hash(wx, wy)
        biome = ch >> 27
        seed = fine_hash(wx, wy)
        expanded = expand(seed)
        r, g, b = seed_tile_color(wx, wy)
        
        print(f"  Tile ({wx},{wy}): biome={BIOME_NAMES[biome]:>14s}  "
              f"seed=0x{seed:08X}  "
              f"expand[{len(expanded)}]={expanded.hex()[:16]}  "
              f"color=#{r:02X}{g:02X}{b:02X}")


if __name__ == "__main__":
    demo()

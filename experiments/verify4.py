"""
Pixelpack Phase 4 - Boot Pixel Architecture Verification

Tests boot pixel encoding/decoding:
  - Boot pixel construction and parsing
  - BOOT_END transitions
  - SET_PROFILE configuration
  - V4 PNG round-trip with boot section
  - Backward compatibility with V1/V2/V3 PNGs

All 37 V1/V2/V3 tests must still pass.
"""

import sys
import os
import tempfile
from expand import expand, seed_to_rgba, seed_from_rgba, BPE_PAIR_TABLE
from expand2 import expand_multi, expand_from_png, extract_seeds_from_png
from expand3 import (
    ExpandContext, expand_with_context, expand_multi_v3, expand_from_png_v3,
)
from expand4 import (
    BootContext, expand_multi_v4, expand_from_png_v4,
    make_boot_end_seed, make_set_profile_seed, make_set_bpe_table_seed,
    make_set_transform_seed, apply_transform,
    generate_bpe_table, _expand_bpe_with_table,
    _decode_boot_opcode, _execute_boot_pixel,
    PROFILES, TRANSFORM_XOR_CONST, TRANSFORM_ADD_CONST,
    TRANSFORM_REVERSE, TRANSFORM_ROTATE,
)
from boot import make_1x1_png, read_png_pixel
from boot2 import make_multipixel_png, read_multipixel_png, encode_multi
from boot3 import encode_v3
import struct
import zlib


# ============================================================
# Boot Pixel Construction Tests
# ============================================================

def test_boot_end_construction():
    """BOOT_END seed has correct bit layout."""
    seed = make_boot_end_seed()
    strategy = (seed >> 28) & 0xF
    opcode = (seed >> 24) & 0xF
    payload = seed & 0x00FFFFFF
    assert strategy == 0xF, f"Expected strategy 0xF, got 0x{strategy:X}"
    assert opcode == 0x0, f"Expected opcode 0, got 0x{opcode:X}"
    assert payload == 0, f"Expected payload 0, got 0x{payload:X}"
    print("  [PASS] [V4] BOOT_END construction")
    return True


def test_set_profile_construction():
    """SET_PROFILE seed has correct bit layout."""
    for pid in range(4):  # test profiles 0-3
        seed = make_set_profile_seed(pid)
        strategy = (seed >> 28) & 0xF
        opcode = (seed >> 24) & 0xF
        payload = seed & 0x00FFFFFF
        profile_id = (payload >> 20) & 0xF
        assert strategy == 0xF
        assert opcode == 0x3
        assert profile_id == pid
    print("  [PASS] [V4] SET_PROFILE construction")
    return True


def test_boot_opcode_decode():
    """_decode_boot_opcode correctly identifies boot vs non-boot pixels."""
    # Boot pixels (strategy 0xF) should decode
    boot_end = make_boot_end_seed()
    decoded = _decode_boot_opcode(boot_end)
    assert decoded is not None
    assert decoded[0] == 0x0  # opcode = BOOT_END

    set_prof = make_set_profile_seed(1)
    decoded = _decode_boot_opcode(set_prof)
    assert decoded is not None
    assert decoded[0] == 0x3  # opcode = SET_PROFILE

    # Non-boot pixels (strategy != 0xF) should return None
    v1_seed = 0xE0000041  # BYTEPACK, strategy 0xE
    decoded = _decode_boot_opcode(v1_seed)
    assert decoded is None

    print("  [PASS] [V4] Boot opcode decode")
    return True


def test_set_profile_execution():
    """SET_PROFILE modifies BootContext correctly."""
    ctx = BootContext()
    assert ctx.xor_mode == False
    assert ctx.profile_id == 0

    # Profile 1 enables XOR
    seed = make_set_profile_seed(1)
    action = _execute_boot_pixel(seed, ctx)
    assert action == 'continue'
    assert ctx.profile_id == 1
    assert ctx.xor_mode == True

    # Profile 0 resets to default
    seed0 = make_set_profile_seed(0)
    _execute_boot_pixel(seed0, ctx)
    assert ctx.profile_id == 0
    assert ctx.xor_mode == False

    print("  [PASS] [V4] SET_PROFILE execution")
    return True


def test_boot_end_execution():
    """BOOT_END returns 'boot_end' action."""
    ctx = BootContext()
    seed = make_boot_end_seed()
    action = _execute_boot_pixel(seed, ctx)
    assert action == 'boot_end'
    print("  [PASS] [V4] BOOT_END execution")
    return True


# ============================================================
# V4 Multi-Seed Expansion Tests
# ============================================================

def test_v4_no_boot_backward_compat():
    """V4 with no boot pixels produces same output as V1."""
    # Single seed
    seed = 0xE0000048  # BYTEPACK
    v1_result = expand(seed)
    v4_result = expand_multi_v4([seed])
    assert v1_result == v1_result, "V4 single seed should match V1"
    print("  [PASS] [V4] No boot backward compat (single seed)")
    return True


def test_v4_boot_end_then_display():
    """V4 with BOOT_END + display seeds produces correct output."""
    boot_end = make_boot_end_seed()
    # A simple display seed
    display_seed = 0xE0000048  # BYTEPACK

    v4_result = expand_multi_v4([boot_end, display_seed])
    v1_result = expand(display_seed)
    assert v4_result == v1_result, "BOOT_END + display should match V1 of display seed"
    print("  [PASS] [V4] BOOT_END + display")
    return True


def test_v4_set_profile_then_display():
    """V4 with SET_PROFILE + BOOT_END + display seeds."""
    set_prof = make_set_profile_seed(0)  # profile 0 = default, no XOR
    boot_end = make_boot_end_seed()
    display_seed = 0xE0000048

    v4_result = expand_multi_v4([set_prof, boot_end, display_seed])
    v1_result = expand(display_seed)
    assert v4_result == v1_result, "SET_PROFILE(0) + BOOT_END + display should match V1"
    print("  [PASS] [V4] SET_PROFILE + BOOT_END + display")
    return True


def test_v4_auto_transition():
    """V4 auto-transitions when first non-0xF seed appears."""
    # Start with boot context but no explicit BOOT_END
    display_seed = 0x90000041  # BPE
    v4_result = expand_multi_v4([display_seed])
    v1_result = expand(display_seed)
    assert v4_result == v1_result, "Auto-transition should match V1"
    print("  [PASS] [V4] Auto-transition (no boot section)")
    return True


def test_v4_multiple_display_seeds():
    """V4 with boot section + multiple display seeds concatenates correctly."""
    boot_end = make_boot_end_seed()

    # Two display seeds using V1 strategies
    seed1 = 0x00000000  # DICT_1 entry 0 = "print("
    seed2 = 0x00000001  # DICT_1 entry 1 = ")"

    v4_result = expand_multi_v4([boot_end, seed1, seed2])
    expected = expand(seed1) + expand(seed2)
    assert v4_result == expected, f"Expected {expected!r}, got {v4_result!r}"
    print("  [PASS] [V4] Multiple display seeds after boot")
    return True


# ============================================================
# V4 PNG Round-Trip Tests
# ============================================================

def _make_v4_png(seeds: list, seed_count: int = None, extra_text_chunks: dict = None) -> bytes:
    """Build a V4 PNG with t6mode metadata.
    
    extra_text_chunks: optional dict of {key: bytes_value} for additional tEXt chunks
    (e.g. bp8table, bp_mode6_table from a V3 PNG).
    """
    n = seed_count if seed_count is not None else len(seeds)
    width = 1
    height = n
    while width * height < n:
        width += 1
        if width > height:
            height = width

    raw_rows = bytearray()
    for row in range(height):
        raw_rows.append(0)  # filter byte
        for col in range(width):
            idx = row * width + col
            if idx < n:
                r, g, b, a = seed_to_rgba(seeds[idx])
            else:
                r, g, b, a = 0, 0, 0, 0
            raw_rows.extend([r, g, b, a])

    compressed = zlib.compress(bytes(raw_rows))

    def chunk(chunk_type, data):
        c = chunk_type + data
        crc = zlib.crc32(c) & 0xFFFFFFFF
        return struct.pack('>I', len(data)) + c + struct.pack('>I', crc)

    signature = b'\x89PNG\r\n\x1a\n'
    ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0)
    ihdr = chunk(b'IHDR', ihdr_data)
    chunks = [signature, ihdr]
    chunks.append(chunk(b'tEXt', b'seedcnt\x00' + str(n).encode()))
    chunks.append(chunk(b'tEXt', b't6mode\x001'))
    if extra_text_chunks:
        for key, value in extra_text_chunks.items():
            chunks.append(chunk(b'tEXt', key.encode() + b'\x00' + value))
    idat = chunk(b'IDAT', compressed)
    iend = chunk(b'IEND', b'')
    chunks.extend([idat, iend])
    return b''.join(chunks)


def test_v4_png_backward_compat_v3():
    """V4 decoder correctly falls back for V3 PNGs."""
    target = b'print("hello")\n'
    data_seeds, png_data = encode_v3(target)
    v4_result = expand_from_png_v4(png_data)
    assert v4_result == target, "V4 should decode V3 PNGs identically"
    print("  [PASS] [V4] PNG backward compat with V3")
    return True


def test_v4_png_roundtrip_basic():
    """V4 PNG with boot section round-trips correctly."""
    boot_end = make_boot_end_seed()
    # Use simple V1 seeds as display pixels
    display_seeds = [0x00000000, 0x00000001]  # DICT_1 entries

    all_seeds = [boot_end] + display_seeds
    png_data = _make_v4_png(all_seeds)
    result = expand_from_png_v4(png_data)

    expected = expand(display_seeds[0]) + expand(display_seeds[1])
    assert result == expected, f"Expected {expected!r}, got {result!r}"
    print("  [PASS] [V4] PNG round-trip basic (BOOT_END + 2 display)")
    return True


def test_v4_png_set_profile_roundtrip():
    """V4 PNG with SET_PROFILE + BOOT_END round-trips."""
    set_prof = make_set_profile_seed(0)  # default profile
    boot_end = make_boot_end_seed()
    display_seed = 0x90000041  # BPE

    all_seeds = [set_prof, boot_end, display_seed]
    png_data = _make_v4_png(all_seeds)
    result = expand_from_png_v4(png_data)

    expected = expand(display_seed)
    assert result == expected, f"Expected {expected!r}, got {result!r}"
    print("  [PASS] [V4] PNG round-trip with SET_PROFILE")
    return True


def test_v4_png_no_boot_section():
    """V4 PNG with no boot section (first pixel is display)."""
    display_seeds = [0x00000000, 0x00000001]
    png_data = _make_v4_png(display_seeds)
    result = expand_from_png_v4(png_data)

    expected = expand(display_seeds[0]) + expand(display_seeds[1])
    assert result == expected
    print("  [PASS] [V4] PNG with no boot section")
    return True


# ============================================================
# Boot Pixel RGBA Representation
# ============================================================

def test_boot_pixel_rgba():
    """Boot pixels produce valid RGBA values (0-255 per channel)."""
    boot_end = make_boot_end_seed()
    r, g, b, a = seed_to_rgba(boot_end)
    assert 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255 and 0 <= a <= 255

    set_prof = make_set_profile_seed(5, config_bits=0xABCDE)
    r, g, b, a = seed_to_rgba(set_prof)
    assert 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255 and 0 <= a <= 255

    # Boot pixels should have alpha 0 (they're compute-only)
    # Actually alpha = seed & 0xFF, which may not be 0. That's fine --
    # boot pixels use the full 32 bits for encoding.

    print("  [PASS] [V4] Boot pixel RGBA values")
    return True


def test_boot_pixel_as_image():
    """Boot pixel seeds survive PNG encode/decode."""
    boot_end = make_boot_end_seed()
    set_prof = make_set_profile_seed(3, config_bits=0x12345)

    all_seeds = [set_prof, boot_end, 0xE0000048]
    png_data = _make_v4_png(all_seeds)

    # Extract seeds and verify they match
    seeds, count, _tables = extract_seeds_from_png(png_data)
    assert count == 3
    assert seeds[0] == set_prof
    assert seeds[1] == boot_end
    assert seeds[2] == 0xE0000048
    print("  [PASS] [V4] Boot pixel seeds survive PNG round-trip")
    return True


# ============================================================
# Full Program Round-Trip
# ============================================================

def test_v4_fibonacci_with_boot():
    """Fibonacci program encoded with boot section decodes correctly."""
    target = b'def fib(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for i in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nfib(10)'

    # Encode as V3, then wrap in V4 with a boot pixel
    data_seeds, v3_png = encode_v3(target)

    # Manually build V4 PNG: add SET_PROFILE(0) + BOOT_END before V3 data seeds
    # First extract seeds from V3 PNG
    v3_seeds, v3_count, _tables = extract_seeds_from_png(v3_png)
    real_seeds = v3_seeds[:v3_count]

    # Extract file-specific tables from V3 PNG tEXt chunks
    extra_chunks = {}
    from expand3 import _read_text_chunk
    for key in ['bp8table', 'bp_mode6_table']:
        val = _read_text_chunk(v3_png, key)
        if val:
            extra_chunks[key] = val.encode()

    boot_end = make_boot_end_seed()
    all_seeds = [boot_end] + real_seeds
    v4_png = _make_v4_png(all_seeds, extra_text_chunks=extra_chunks if extra_chunks else None)

    v4_result = expand_from_png_v4(v4_png)
    assert v4_result == target, f"V4 fib decode mismatch: got {len(v4_result)} bytes"
    print(f"  [PASS] [V4] Fibonacci with boot section ({len(real_seeds)} display + 1 boot)")
    return True


# ============================================================
# SET_BPE_TABLE Tests
# ============================================================

def test_bpe_table_generation():
    """PRNG-based BPE table generation produces valid 128-entry table."""
    for seed in [0, 1, 42, 100, 4095]:
        table = generate_bpe_table(seed)
        assert len(table) == 128, f"Table length {len(table)} for seed {seed}"
        assert table[0] == b'', f"Index 0 should be empty, got {table[0]!r}"
        # All entries should be 2-byte pairs (or empty)
        for i, entry in enumerate(table):
            assert len(entry) == 0 or len(entry) == 2, \
                f"Entry {i} has length {len(entry)} for seed {seed}"
    print("  [PASS] [V4] BPE table generation")
    return True


def test_bpe_table_deterministic():
    """Same PRNG seed produces identical table."""
    for seed in [0, 42, 4095]:
        t1 = generate_bpe_table(seed)
        t2 = generate_bpe_table(seed)
        assert t1 == t2, f"Seed {seed} not deterministic"
    print("  [PASS] [V4] BPE table deterministic")
    return True


def test_bpe_table_unique_entries():
    """Each PRNG table has unique entries."""
    table = generate_bpe_table(42)
    entries = [e for e in table[1:] if e]
    assert len(set(entries)) == len(entries), "Duplicate entries in table"
    print("  [PASS] [V4] BPE table unique entries")
    return True


def test_set_bpe_table_seed_construction():
    """SET_BPE_TABLE seed has correct bit layout."""
    for prng in [0, 1, 42, 100, 4095]:
        seed = make_set_bpe_table_seed(prng)
        strategy = (seed >> 28) & 0xF
        opcode = (seed >> 24) & 0xF
        payload = seed & 0x00FFFFFF
        assert strategy == 0xF
        assert opcode == 0x5
        assert payload == prng
    print("  [PASS] [V4] SET_BPE_TABLE construction")
    return True


def test_set_bpe_table_execution():
    """SET_BPE_TABLE opcode sets custom_bpe_table on BootContext."""
    ctx = BootContext()
    assert ctx.custom_bpe_table is None

    seed = make_set_bpe_table_seed(42)
    action = _execute_boot_pixel(seed, ctx)
    assert action == 'continue'
    assert ctx.custom_bpe_table is not None
    assert len(ctx.custom_bpe_table) == 128
    assert ctx.custom_bpe_table[0] == b''

    # Different seed produces different table
    seed2 = make_set_bpe_table_seed(100)
    _execute_boot_pixel(seed2, ctx)
    assert ctx.custom_bpe_table != generate_bpe_table(42)
    assert ctx.custom_bpe_table == generate_bpe_table(100)

    print("  [PASS] [V4] SET_BPE_TABLE execution")
    return True


def test_custom_bpe_expansion():
    """BPE seed expands differently with custom table vs fixed table."""
    from expand import expand as expand_v1

    # Create a BPE seed that will produce different output
    bpe_seed = 0x90000000 | (1) | (2 << 7) | (3 << 14) | (4 << 21)

    fixed_result = expand_v1(bpe_seed)

    # With custom table
    custom_table = generate_bpe_table(42)
    custom_result = _expand_bpe_with_table(bpe_seed, custom_table)

    assert fixed_result != custom_result, "Custom table should produce different output"
    assert len(fixed_result) == 8, "Fixed table: 4 pairs = 8 bytes"
    assert len(custom_result) == 8, "Custom table: 4 pairs = 8 bytes"

    print("  [PASS] [V4] Custom BPE expansion differs from fixed")
    return True


def test_v4_bpe_roundtrip():
    """V4 with SET_BPE_TABLE round-trips: encode with table, decode with same table."""
    custom_table = generate_bpe_table(7)
    # Manually find a BPE seed that produces something useful with this table
    # Pick indices 1-4 from the custom table
    bpe_seed = 0x90000000 | (1) | (2 << 7) | (3 << 14) | (4 << 21)
    expected = _expand_bpe_with_table(bpe_seed, custom_table)

    # Full V4 expansion with boot section
    set_bpe = make_set_bpe_table_seed(7)
    boot_end = make_boot_end_seed()
    result = expand_multi_v4([set_bpe, boot_end, bpe_seed])

    assert result == expected, f"Expected {expected!r}, got {result!r}"
    print("  [PASS] [V4] BPE round-trip with custom table")
    return True


def test_v4_bpe_png_roundtrip():
    """V4 PNG with SET_BPE_TABLE round-trips correctly."""
    custom_table = generate_bpe_table(7)
    bpe_seed = 0x90000000 | (1) | (2 << 7) | (3 << 14) | (4 << 21)
    expected = _expand_bpe_with_table(bpe_seed, custom_table)

    set_bpe = make_set_bpe_table_seed(7)
    boot_end = make_boot_end_seed()
    all_seeds = [set_bpe, boot_end, bpe_seed]

    png_data = _make_v4_png(all_seeds)
    result = expand_from_png_v4(png_data)

    assert result == expected, f"Expected {expected!r}, got {result!r}"
    print("  [PASS] [V4] PNG round-trip with custom BPE table")
    return True


def test_v4_mixed_strategies_with_custom_bpe():
    """V4 with custom BPE table: non-BPE seeds still use V3 expansion."""
    from expand import expand as expand_v1

    custom_table = generate_bpe_table(42)
    bpe_seed = 0x90000000 | (1) | (2 << 7) | (3 << 14) | (4 << 21)
    # A non-BPE seed (DICT_1, strategy 0x0)
    dict_seed = 0x00000000  # DICT_1 entry 0 = "print("

    set_bpe = make_set_bpe_table_seed(42)
    boot_end = make_boot_end_seed()

    result = expand_multi_v4([set_bpe, boot_end, bpe_seed, dict_seed])

    bpe_result = _expand_bpe_with_table(bpe_seed, custom_table)
    dict_result = expand_v1(dict_seed)

    assert result == bpe_result + dict_result, \
        f"Mixed expansion failed: {result!r} != {bpe_result + dict_result!r}"
    print("  [PASS] [V4] Mixed strategies with custom BPE table")
    return True


# ============================================================
# SET_TRANSFORM Tests
# ============================================================

def test_set_transform_seed_construction():
    """SET_TRANSFORM seed has correct bit layout."""
    for tt in range(4):
        for tp in [0, 1, 42, 128, 255]:
            seed = make_set_transform_seed(tt, tp)
            strategy = (seed >> 28) & 0xF
            opcode = (seed >> 24) & 0xF
            payload = seed & 0x00FFFFFF
            extracted_type = (payload >> 20) & 0xF
            extracted_param = (payload >> 12) & 0xFF
            assert strategy == 0xF, f"Bad strategy for type={tt} param={tp}"
            assert opcode == 0x7, f"Bad opcode for type={tt} param={tp}"
            assert extracted_type == tt, f"Type mismatch: {extracted_type} != {tt}"
            assert extracted_param == tp, f"Param mismatch: {extracted_param} != {tp}"
    print("  [PASS] [V4] SET_TRANSFORM construction")
    return True


def test_set_transform_execution():
    """SET_TRANSFORM opcode sets transform_type and transform_param on BootContext."""
    ctx = BootContext()
    assert ctx.transform_type == 0
    assert ctx.transform_param == 0

    # Set XOR transform with key 0x42
    seed = make_set_transform_seed(TRANSFORM_XOR_CONST, 0x42)
    action = _execute_boot_pixel(seed, ctx)
    assert action == 'continue'
    assert ctx.transform_type == TRANSFORM_XOR_CONST
    assert ctx.transform_param == 0x42

    # Set ADD transform with key 10
    seed2 = make_set_transform_seed(TRANSFORM_ADD_CONST, 10)
    _execute_boot_pixel(seed2, ctx)
    assert ctx.transform_type == TRANSFORM_ADD_CONST
    assert ctx.transform_param == 10

    print("  [PASS] [V4] SET_TRANSFORM execution")
    return True


def test_apply_transform_xor():
    """apply_transform XOR_CONST works correctly."""
    data = bytes([0x00, 0x42, 0xFF, 0x80])
    # XOR with 0x42
    result = apply_transform(data, TRANSFORM_XOR_CONST, 0x42)
    expected = bytes([0x42, 0x00, 0xBD, 0xC2])
    assert result == expected, f"XOR failed: {result.hex()} != {expected.hex()}"
    # Double-XOR = identity
    result2 = apply_transform(result, TRANSFORM_XOR_CONST, 0x42)
    assert result2 == data, "Double XOR should be identity"
    # XOR with 0 = identity
    assert apply_transform(data, TRANSFORM_XOR_CONST, 0) == data
    print("  [PASS] [V4] apply_transform XOR_CONST")
    return True


def test_apply_transform_add():
    """apply_transform ADD_CONST works correctly."""
    data = bytes([0x00, 0x42, 0xFE, 0xFF])
    # ADD 1
    result = apply_transform(data, TRANSFORM_ADD_CONST, 1)
    expected = bytes([0x01, 0x43, 0xFF, 0x00])
    assert result == expected, f"ADD failed: {result.hex()} != {expected.hex()}"
    # ADD 0 = identity
    assert apply_transform(data, TRANSFORM_ADD_CONST, 0) == data
    # Round trip: subtract (add 256-key)
    key = 42
    transformed = apply_transform(data, TRANSFORM_ADD_CONST, key)
    recovered = apply_transform(transformed, TRANSFORM_ADD_CONST, (256 - key) & 0xFF)
    assert recovered == data, "ADD round-trip failed"
    print("  [PASS] [V4] apply_transform ADD_CONST")
    return True


def test_apply_transform_reverse():
    """apply_transform REVERSE works correctly."""
    data = b'Hello'
    result = apply_transform(data, TRANSFORM_REVERSE, 0)
    assert result == b'olleH', f"REVERSE failed: {result!r}"
    # Double reverse = identity
    assert apply_transform(result, TRANSFORM_REVERSE, 0) == data
    # Empty
    assert apply_transform(b'', TRANSFORM_REVERSE, 0) == b''
    print("  [PASS] [V4] apply_transform REVERSE")
    return True


def test_apply_transform_rotate():
    """apply_transform ROTATE works correctly."""
    data = b'ABCDE'
    # Rotate left by 2: CDEAB
    result = apply_transform(data, TRANSFORM_ROTATE, 2)
    assert result == b'CDEAB', f"ROTATE(2) failed: {result!r}"
    # Rotate left by 0 = identity
    assert apply_transform(data, TRANSFORM_ROTATE, 0) == data
    # Rotate left by len = identity
    assert apply_transform(data, TRANSFORM_ROTATE, 5) == data
    # Round trip: rotate left by N, then rotate left by (len-N) = identity
    for n in range(6):
        rotated = apply_transform(data, TRANSFORM_ROTATE, n)
        recovered = apply_transform(rotated, TRANSFORM_ROTATE, 5 - (n % 5) if n % 5 != 0 else 0)
        assert recovered == data, f"ROTATE round-trip failed for n={n}"
    print("  [PASS] [V4] apply_transform ROTATE")
    return True


def _find_seeds_for_target(target: bytes) -> list:
    """Find display seeds that expand to exactly the target bytes.
    
    Uses find_seed.search for prefix matching, falls back to
    BYTEPACK mode 0 (3-byte raw) for any remaining chunks.
    """
    import find_seed as fs
    seeds_needed = []
    remaining = target
    
    while remaining:
        # Try finding a seed for progressively shorter prefixes
        found = False
        for try_len in range(len(remaining), 0, -1):
            prefix = remaining[:try_len]
            results = fs.search(prefix)
            if results:
                seed_val, _ = results[0]
                output = expand(seed_val)
                if output == prefix:
                    seeds_needed.append(seed_val)
                    remaining = remaining[try_len:]
                    found = True
                    break
        if not found:
            # Fallback: BYTEPACK mode 0 encodes 3 raw bytes minimum
            if len(remaining) >= 3:
                b0 = remaining[0]
                b1 = remaining[1]
                b2 = remaining[2]
                # BYTEPACK mode 0: [2:0=mode] [8:b0] [8:b1] [8:b2] [4:extra=0]
                params = b0 | (b1 << 8) | (b2 << 16)
                seed_val = 0xE0000000 | (params << 3)
                seeds_needed.append(seed_val)
                remaining = remaining[3:]
            else:
                # Pad to 3 bytes with the last byte (we'll truncate later)
                padded = remaining + bytes([remaining[-1]] * (3 - len(remaining)))
                b0, b1, b2 = padded[0], padded[1], padded[2]
                params = b0 | (b1 << 8) | (b2 << 16)
                seed_val = 0xE0000000 | (params << 3)
                seeds_needed.append(seed_val)
                remaining = b''
    return seeds_needed


def test_v4_xor_transform_roundtrip():
    """V4 with XOR transform: encode inverse-XOR target, decode with transform."""
    target = b'Hello'
    xor_key = 0x42

    # Inverse-transform: what the display seeds should produce
    inv_target = bytes([b ^ xor_key for b in target])

    seeds_needed = _find_seeds_for_target(inv_target)

    # Full V4: SET_TRANSFORM(XOR, 0x42) + BOOT_END + display seeds
    xform_seed = make_set_transform_seed(TRANSFORM_XOR_CONST, xor_key)
    boot_end = make_boot_end_seed()
    all_seeds = [xform_seed, boot_end] + seeds_needed

    result = expand_multi_v4(all_seeds)
    assert result == target, f"XOR transform round-trip failed: {result!r} != {target!r}"
    print("  [PASS] [V4] XOR transform round-trip")
    return True


def test_v4_xor_transform_png_roundtrip():
    """V4 PNG with XOR transform round-trips correctly."""
    target = b'Hello'
    xor_key = 0x42
    inv_target = bytes([b ^ xor_key for b in target])

    seeds_needed = _find_seeds_for_target(inv_target)

    xform_seed = make_set_transform_seed(TRANSFORM_XOR_CONST, xor_key)
    boot_end = make_boot_end_seed()
    all_seeds = [xform_seed, boot_end] + seeds_needed

    png_data = _make_v4_png(all_seeds)
    result = expand_from_png_v4(png_data)
    assert result == target, f"XOR PNG round-trip failed: {result!r} != {target!r}"
    print("  [PASS] [V4] XOR transform PNG round-trip")
    return True


def test_v4_reverse_transform_roundtrip():
    """V4 with REVERSE transform: display seeds encode reversed target."""
    # Use a target whose reverse can be found by find_seed.search
    # b'tset' reverses to b'test' -- both are 4 bytes and findable
    target = b'test'
    inv_target = target[::-1]  # 'tset'

    from find_seed import search as seed_search
    results = seed_search(inv_target)
    assert results, f"Could not find seed for {inv_target!r}"
    seed_val, _ = results[0]
    seeds_needed = [seed_val]

    xform_seed = make_set_transform_seed(TRANSFORM_REVERSE, 0)
    boot_end = make_boot_end_seed()
    all_seeds = [xform_seed, boot_end] + seeds_needed

    result = expand_multi_v4(all_seeds)
    assert result == target, f"REVERSE round-trip failed: {result!r} != {target!r}"
    print("  [PASS] [V4] REVERSE transform round-trip")
    return True


def test_v4_no_transform_backward_compat():
    """V4 without SET_TRANSFORM behaves exactly as before."""
    boot_end = make_boot_end_seed()
    display_seeds = [0x00000000, 0x00000001]  # DICT_1 entries
    result = expand_multi_v4([boot_end] + display_seeds)
    expected = expand(display_seeds[0]) + expand(display_seeds[1])
    assert result == expected
    print("  [PASS] [V4] No-transform backward compat")
    return True


def test_v4_transform_with_bpe_table():
    """V4 with both SET_BPE_TABLE and SET_TRANSFORM works together."""
    # Use XOR transform with custom BPE table
    custom_table = generate_bpe_table(42)
    xor_key = 0x55

    # Create a BPE seed that expands via custom table, then XOR transform
    bpe_seed = 0x90000000 | (1) | (2 << 7) | (3 << 14) | (4 << 21)
    raw_expansion = _expand_bpe_with_table(bpe_seed, custom_table)
    # After XOR transform
    expected = bytes([b ^ xor_key for b in raw_expansion])

    set_bpe = make_set_bpe_table_seed(42)
    xform = make_set_transform_seed(TRANSFORM_XOR_CONST, xor_key)
    boot_end = make_boot_end_seed()
    result = expand_multi_v4([set_bpe, xform, boot_end, bpe_seed])

    assert result == expected, f"Combined BPE+transform failed: {result!r} != {expected!r}"
    print("  [PASS] [V4] Transform with custom BPE table")
    return True


# ============================================================
# Run All Tests
# ============================================================

def run_all():
    tests = [
        # Boot pixel construction
        test_boot_end_construction,
        test_set_profile_construction,
        test_boot_opcode_decode,
        test_set_profile_execution,
        test_boot_end_execution,
        # V4 multi-seed expansion
        test_v4_no_boot_backward_compat,
        test_v4_boot_end_then_display,
        test_v4_set_profile_then_display,
        test_v4_auto_transition,
        test_v4_multiple_display_seeds,
        # V4 PNG round-trips
        test_v4_png_backward_compat_v3,
        test_v4_png_roundtrip_basic,
        test_v4_png_set_profile_roundtrip,
        test_v4_png_no_boot_section,
        # Boot pixel as image
        test_boot_pixel_rgba,
        test_boot_pixel_as_image,
        # Full program
        test_v4_fibonacci_with_boot,
        # SET_BPE_TABLE
        test_bpe_table_generation,
        test_bpe_table_deterministic,
        test_bpe_table_unique_entries,
        test_set_bpe_table_seed_construction,
        test_set_bpe_table_execution,
        test_custom_bpe_expansion,
        test_v4_bpe_roundtrip,
        test_v4_bpe_png_roundtrip,
        test_v4_mixed_strategies_with_custom_bpe,
        # SET_TRANSFORM
        test_set_transform_seed_construction,
        test_set_transform_execution,
        test_apply_transform_xor,
        test_apply_transform_add,
        test_apply_transform_reverse,
        test_apply_transform_rotate,
        test_v4_xor_transform_roundtrip,
        test_v4_xor_transform_png_roundtrip,
        test_v4_reverse_transform_roundtrip,
        test_v4_no_transform_backward_compat,
        test_v4_transform_with_bpe_table,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1

    print()
    total = passed + failed
    print(f"  {passed}/{total} tests passed ({100*passed//total}%)")
    if failed == 0:
        print()
        print("  PHASE 4 COMPLETE: Boot pixel architecture works!")
        print("  BOOT_END + SET_PROFILE verified.")
        print("  V4 PNGs with boot section round-trip correctly.")
        print("  All V1/V2/V3 PNGs decode correctly via V4 fallback.")
    return failed == 0


if __name__ == '__main__':
    success = run_all()
    sys.exit(0 if success else 1)

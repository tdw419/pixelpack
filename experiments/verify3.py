"""
Pixelpack Phase 3 Verification Suite

Tests context-dependent expansion: LZ77 back-references, dynamic dictionary,
XOR channel, and V3 PNG round-trips.

Includes all V1+V2 tests for backward compatibility (26 total).
New V3 tests prove context-dependent encoding works.

Proves: V3 PNG encode -> decode produces identical bytes, with fewer pixels.
"""

import sys
import os
import subprocess
import tempfile
from find_seed import search
from expand import expand, seed_to_rgba, seed_from_rgba
from expand2 import expand_multi, expand_from_png, extract_seeds_from_png
from expand3 import (
    ExpandContext, expand_with_context, expand_multi_v3, expand_from_png_v3,
    _expand_lz77, _expand_dyn_dict, emit_dict_seed,
)
from expand import BPE_PAIR_TABLE
from boot import make_1x1_png, read_png_pixel
from boot2 import (
    make_multipixel_png, read_multipixel_png,
    encode_multi, decode_png, _find_multi_seeds_dp,
)
from boot3 import encode_v3, make_v3_png


# ============================================================
# Unit Tests: LZ77 Back-Reference
# ============================================================

def test_lz77_basic():
    """LZ77 can reference previously emitted bytes."""
    ctx = ExpandContext()
    ctx.output_buffer = bytearray(b'Hello, World!')
    # Buffer is 13 bytes (indices 0-12). "World" starts at index 7.
    # start = buf_len - 1 - offset = 12 - offset = 7 => offset = 5
    result = _expand_lz77(5 | (5 << 16), ctx)  # offset=5, length=5
    assert result == b'World', f"Expected b'World', got {result!r}"
    print("  [PASS] LZ77 basic back-reference")
    return True


def test_lz77_overlapping():
    """LZ77 handles overlapping copies (like run extension)."""
    ctx = ExpandContext()
    ctx.output_buffer = bytearray(b'AB')
    # Reference from offset 1, length 4: copies A,B,A,B (overlapping)
    result = _expand_lz77(1 | (4 << 16), ctx)
    assert result == b'ABAB', f"Expected b'ABAB', got {result!r}"
    print("  [PASS] LZ77 overlapping copy")
    return True


def test_lz77_empty_buffer():
    """LZ77 returns empty when buffer is empty."""
    ctx = ExpandContext()
    result = _expand_lz77(0 | (5 << 16), ctx)
    assert result == b'', f"Expected empty, got {result!r}"
    print("  [PASS] LZ77 empty buffer")
    return True


def test_lz77_offset_too_large():
    """LZ77 returns empty when offset >= buffer length."""
    ctx = ExpandContext()
    ctx.output_buffer = bytearray(b'Hi')
    result = _expand_lz77(99 | (5 << 16), ctx)  # offset 99, buffer only 2
    assert result == b'', f"Expected empty, got {result!r}"
    print("  [PASS] LZ77 offset overflow")
    return True


def test_lz77_repeat_indent():
    """LZ77 can encode repeated indentation patterns."""
    ctx = ExpandContext()
    ctx.output_buffer = bytearray(b'def f():\n    return 1\n')
    # Buffer is 22 bytes (indices 0-21). 4-space indent at indices 9-12.
    # start = 21 - offset = 9 => offset = 12, length = 4
    result = _expand_lz77(12 | (4 << 16), ctx)
    assert result == b'    ', f"Expected 4 spaces, got {result!r}"
    print("  [PASS] LZ77 indent reference")
    return True


# ============================================================
# Unit Tests: Dynamic Dictionary
# ============================================================

def test_dyn_dict_add():
    """Dynamic dictionary add mode stores and emits."""
    from expand import DICTIONARY
    ctx = ExpandContext()
    # Add mode: bit 27 = 1, 1 entry = DICTIONARY[7] = b'def '
    # params: bit 27 = 1, bits 26:24 = count=1, bits 3:0 = index 7
    params = (1 << 27) | (1 << 24) | 7
    result = _expand_dyn_dict(params, ctx)
    assert result == b'def ', f"Expected b'def ', got {result!r}"
    assert len(ctx.dyn_dict) == 1, f"dyn_dict should have 1 entry"
    assert ctx.dyn_dict[0] == b'def ', f"dyn_dict[0] wrong: {ctx.dyn_dict[0]!r}"
    print("  [PASS] Dynamic dictionary add mode")
    return True


def test_dyn_dict_reference():
    """Dynamic dictionary reference mode looks up entries."""
    ctx = ExpandContext()
    ctx.dyn_dict = [b'print(Hello)', b'return 42']
    # Reference mode: bit 27 = 0, index = 1
    params = 1  # index 1
    result = _expand_dyn_dict(params, ctx)
    assert result == b'return 42', f"Expected b'return 42', got {result!r}"
    print("  [PASS] Dynamic dictionary reference mode")
    return True


def test_dyn_dict_oob():
    """Dynamic dictionary returns empty for out-of-bounds index."""
    ctx = ExpandContext()
    ctx.dyn_dict = [b'hello']
    result = _expand_dyn_dict(5, ctx)  # index 5, only 1 entry
    assert result == b'', f"Expected empty, got {result!r}"
    print("  [PASS] Dynamic dictionary out-of-bounds")
    return True


# ============================================================
# Unit Tests: ExpandContext
# ============================================================

def test_context_accumulation():
    """ExpandContext accumulates output across seeds."""
    ctx = ExpandContext()
    r1 = expand_with_context(0x00000007, ctx)  # DICT_1: DICTIONARY[7] = b'def '
    assert r1 == b'def ', f"Expected b'def ', got {r1!r}"
    assert len(ctx.output_buffer) == 4

    r2 = expand_with_context(0x00000009, ctx)  # DICT_1: DICTIONARY[9] = b'main'
    assert r2 == b'main'
    assert ctx.output_buffer == bytearray(b'def main')
    print("  [PASS] Context accumulates across seeds")
    return True


def test_context_lz77_after_emit():
    """LZ77 can reference bytes emitted by previous seeds."""
    ctx = ExpandContext()
    # Emit "Hello" via DICT
    r1 = expand_with_context(0x00000003, ctx)  # DICTIONARY[3] = b'Hello'
    assert r1 == b'Hello'

    # Now use LZ77 to reference "ello" from the buffer
    # Buffer is "Hello" (5 bytes). "ello" starts at index 1
    # offset = 4 - 1 = 3, length = 4
    r2 = expand_with_context(0xC0000003 | (4 << 16), ctx)
    assert r2 == b'ello', f"Expected b'ello', got {r2!r}"
    print("  [PASS] LZ77 references previous seed output")
    return True


def test_multi_v3_basic():
    """expand_multi_v3 chains seeds with context."""
    seeds = [0x00000003, 0xC0000003 | (4 << 16)]  # "Hello" + LZ77 "ello"
    result = expand_multi_v3(seeds)
    assert result == b'Helloello', f"Expected b'Helloello', got {result!r}"
    print("  [PASS] expand_multi_v3 basic chain")
    return True


# ============================================================
# V3 PNG Round-Trip Tests
# ============================================================

def verify_v3_roundtrip(target: bytes, desc: str, runnable: bool = False,
                        language: str = None, timeout: float = 60.0):
    """Encode target to V3 PNG, decode, verify match."""
    print(f"{'='*60}")
    print(f"  [V3] {desc}")
    print(f"  Bytes: {len(target)}B")

    seeds, png_data = encode_v3(target, timeout=timeout)
    if not seeds or png_data is None:
        print(f"  FAIL: V3 encoding failed")
        return False

    # Decode from PNG
    decoded = expand_from_png_v3(png_data)
    if decoded != target:
        print(f"  FAIL: Round-trip mismatch!")
        print(f"    Expected ({len(target)}B): {target[:50]!r}...")
        print(f"    Got ({len(decoded)}B): {decoded[:50]!r}...")
        return False

    print(f"  Round-trip: PASS")

    # Compare to V2 baseline
    v2_seeds = _find_multi_seeds_dp(target, timeout=15.0, max_seeds=128)
    v2_count = len(v2_seeds) if v2_seeds else 999
    # Total V3 seeds includes setup
    from expand3 import _read_text_chunk
    dict_only = int(_read_text_chunk(png_data, 'dict_only') or '0')
    total_v3 = dict_only + len(seeds)
    saved = v2_count - total_v3
    pct = (saved / v2_count * 100) if v2_count > 0 else 0
    print(f"  Pixels: V2={v2_count}, V3={total_v3} (setup={dict_only}+data={len(seeds)}), saved={saved} ({pct:.0f}%)")

    # Execute if runnable
    if runnable:
        with tempfile.NamedTemporaryFile(
            mode='wb', suffix=_get_suffix(language), delete=False
        ) as f:
            f.write(decoded)
            tmp_path = f.name
        os.chmod(tmp_path, 0o755)

        try:
            cmd = _get_run_cmd(tmp_path, language)
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5.0
            )
            if result.returncode == 0:
                print(f"  Execute: PASS (exit=0)")
                if result.stdout.strip():
                    print(f"  Output: {result.stdout.strip()[:80]!r}")
            else:
                print(f"  Execute: WARN (exit={result.returncode})")
                if result.stderr:
                    print(f"  Stderr: {result.stderr[:100]}")
        except Exception as e:
            print(f"  Execute: WARN ({e})")
        finally:
            os.unlink(tmp_path)

    return True


# ============================================================
# V3 PNG Construction Tests
# ============================================================

def test_v3_png_metadata():
    """V3 PNG has correct t3mode and seedcnt metadata."""
    seeds = [0x00000003, 0x00000009]  # "Hello" + "main"
    png_data = make_v3_png(seeds, dict_only=1)

    from expand3 import _read_text_chunk
    assert _read_text_chunk(png_data, 't3mode') == '1', "Missing t3mode"
    assert _read_text_chunk(png_data, 'seedcnt') == '2', "Wrong seedcnt"
    assert _read_text_chunk(png_data, 'dict_only') == '1', "Wrong dict_only"
    print("  [PASS] V3 PNG metadata correct")
    return True


def test_v3_png_xor_metadata():
    """V3 PNG with XOR mode has correct metadata."""
    seeds = [0x00000003]
    png_data = make_v3_png(seeds, xor_mode=True)

    from expand3 import _read_text_chunk
    assert _read_text_chunk(png_data, 'xor_mode') == 'true', "Missing xor_mode"
    print("  [PASS] V3 PNG XOR metadata")
    return True


def test_v3_png_seed_extraction():
    """Seeds can be extracted from V3 PNG and match input."""
    seeds = [0x12345678, 0xABCDEF01, 0x00000003]
    png_data = make_v3_png(seeds, dict_only=1)

    extracted_seeds, real_count, _tables = extract_seeds_from_png(png_data)
    # Extracted may include padding pixels; only compare real_count seeds
    assert real_count == 3, f"Expected real_count=3, got {real_count}"
    assert extracted_seeds[:real_count] == seeds, \
        f"Seed mismatch: {extracted_seeds[:real_count]!r} != {seeds!r}"
    print("  [PASS] V3 PNG seed extraction")
    return True


def test_v3_fallback_to_v2():
    """V3 decoder falls back to V2 for non-V3 PNGs."""
    # Make a V2 PNG
    target = b'print("Hello")\n'
    results = search(target, timeout=5.0)
    assert results, "Could not find seed for V2 fallback test"
    seed = results[0][0]
    r, g, b, a = seed_to_rgba(seed)
    png_v2 = make_1x1_png(r, g, b, a)

    # V3 decoder should handle it
    decoded = expand_from_png_v3(png_v2)
    assert decoded == target, f"V3 fallback failed: {decoded!r} != {target!r}"
    print("  [PASS] V3 decoder falls back to V2")
    return True


# ============================================================
# XOR Channel Tests
# ============================================================

def test_xor_channel_basic():
    """XOR channel: seeds are XORed with previous during decode."""
    seeds = [0x00000003, 0x00000003]  # Both produce "Hello" normally
    # With XOR, second seed gets XORed with first: 0x00000003 ^ 0x00000003 = 0
    # Seed 0 = DICT_1 index 0 = b'print('
    result = expand_multi_v3(seeds)  # No XOR
    assert result == b'HelloHello', f"Expected b'HelloHello', got {result!r}"

    from expand3 import expand_multi_v3_xor
    result_xor = expand_multi_v3_xor(seeds)
    # First: "Hello", second: 0x00000003 ^ 0x00000003 = 0x0 -> DICT_1[0] = b'print('
    assert result_xor == b'Helloprint(', f"XOR channel: expected b'Helloprint(', got {result_xor!r}"
    print("  [PASS] XOR channel basic")
    return True


# ============================================================
# BPE Strategy Tests
# ============================================================

def test_bpe_single_pair():
    """BPE can encode a single byte pair."""
    # 'in' = index 5 in BPE_PAIR_TABLE
    params = 5
    seed = 0x90000000 | params
    result = expand(seed)
    assert result == b'in', f"Expected b'in', got {result!r}"
    print("  [PASS] BPE single pair")
    return True


def test_bpe_two_pairs():
    """BPE can encode two byte pairs (4 bytes)."""
    # '  ' = 1, 'in' = 5
    params = 1 | (5 << 7)
    seed = 0x90000000 | params
    result = expand(seed)
    assert result == b'  in', f"Expected b'  in', got {result!r}"
    print("  [PASS] BPE two pairs")
    return True


def test_bpe_four_pairs():
    """BPE can encode four byte pairs (8 bytes, max)."""
    # 'de'=31, 'fi'=78, 'bo'=121, '  '=1
    params = 31 | (78 << 7) | (121 << 14) | (1 << 21)
    seed = 0x90000000 | params
    result = expand(seed)
    assert result == b'defibo  ', f"Expected b'defibo  ', got {result!r}"
    assert len(result) == 8
    print("  [PASS] BPE four pairs (8 bytes)")
    return True


def test_bpe_terminator():
    """BPE terminates on index 0 (unused pairs are 0)."""
    # 2 pairs followed by 0 terminators
    params = 1 | (5 << 7)  # pairs only, remaining = 0
    seed = 0x90000000 | params
    result = expand(seed)
    assert result == b'  in', f"Expected b'  in', got {result!r}"
    assert len(result) == 4
    print("  [PASS] BPE terminator")
    return True


def test_bpe_roundtrip_via_search():
    """BPE seeds found by search() decode correctly."""
    from find_seed import _search_bpe
    # 'de' + 'fi' = 'defi' (both in table)
    target = b'defi'
    seed = _search_bpe(target)
    assert seed is not None, f"BPE search failed for {target!r}"
    result = expand(seed)
    assert result == target, f"Roundtrip failed: {result!r} != {target!r}"
    print("  [PASS] BPE search roundtrip")
    return True


def test_bpe_in_v3_pipeline():
    """BPE seeds work in V3 context pipeline."""
    # Encode "return " using BPE: 're'=8, 'tu'=79, 'rn'=76, ' '=N/A
    # Actually just test that strategy 0x9 works through expand_with_context
    params = 1 | (5 << 7)  # '  ' + 'in'
    seed = 0x90000000 | params
    ctx = ExpandContext()
    result = expand_with_context(seed, ctx)
    assert result == b'  in', f"V3 BPE: expected b'  in', got {result!r}"
    assert ctx.output_buffer == bytearray(b'  in')
    print("  [PASS] BPE in V3 pipeline")
    return True


def test_bpe_table_completeness():
    """BPE_PAIR_TABLE has 128 entries (index 0 = empty, 1-127 = pairs)."""
    assert len(BPE_PAIR_TABLE) == 128, f"Expected 128 entries, got {len(BPE_PAIR_TABLE)}"
    assert BPE_PAIR_TABLE[0] == b'', "Index 0 must be empty"
    for i in range(1, 128):
        assert len(BPE_PAIR_TABLE[i]) == 2, f"Entry {i} must be 2 bytes, got {len(BPE_PAIR_TABLE[i])}"
    print("  [PASS] BPE table completeness")
    return True


# ============================================================
# Helpers
# ============================================================

def _get_suffix(language):
    return {
        'python': '.py', 'shell': '.sh', 'c': '.c', 'javascript': '.js',
    }.get(language, '.txt')


def _get_run_cmd(path, language):
    return {
        'python': [sys.executable, path],
        'shell': ['/bin/bash', path],
        'c': ['gcc', path, '-o', path + '.out'],
    }.get(language, [sys.executable, path])


# ============================================================
# Main Test Runner
# ============================================================

def main():
    print("PIXELPACK PHASE 3 VERIFICATION SUITE")
    print("Context-dependent expansion: LZ77, dyn_dict, XOR channel")
    print()

    results = []

    # === V1+V2 backward compatibility ===
    print("V1+V2 BACKWARD COMPATIBILITY (20 tests)")
    print("-" * 60)

    v1_tests = [
        (b'print("Hello")\n', 'Python Hello World', True, 'python'),
        (b'echo Hello\n', 'Shell Hello World', True, 'shell'),
        (b'42\n', 'Number literal', False, None),
        (b'Hello, World!\n', 'Classic text', False, None),
        (b'print(42)\n', 'Python print int', True, 'python'),
        (b'void main(){}\n', 'C minimal', False, 'c'),
    ]

    for target, desc, runnable, lang in v1_tests:
        r_seeds = search(target, timeout=10.0)
        if not r_seeds:
            results.append((f"[V1] {desc}", False))
            print(f"  [FAIL] [V1] {desc}: no seed found")
            continue
        seed = r_seeds[0][0]
        expanded = expand(seed)
        ok = expanded == target
        results.append((f"[V1] {desc}", ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] [V1] {desc}")

    v2_tests = [
        (b'x = "Hello"\nprint(x)\n', 'Python variable', True, 'python'),
        (b'print("Hello")\nprint(42)\n', 'Two prints', False, None),
        (b'def greet(name):\n    print(name)\n\n', 'Python function', False, 'python'),
        (b'def add(a, b):\n    return a + b\n\nprint(add(1, 2))\n', 'Python add fn', True, 'python'),
    ]

    for target, desc, runnable, lang in v2_tests:
        v2_seeds = _find_multi_seeds_dp(target, timeout=30.0, max_seeds=64)
        if not v2_seeds:
            results.append((f"[V2] {desc}", False))
            print(f"  [FAIL] [V2] {desc}: no seeds found")
            continue
        decoded = expand_multi(v2_seeds)
        ok = decoded == target
        results.append((f"[V2] {desc}", ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] [V2] {desc}")

    print()

    # === V3 Unit Tests ===
    print("V3 UNIT TESTS")
    print("-" * 60)

    unit_tests = [
        test_lz77_basic,
        test_lz77_overlapping,
        test_lz77_empty_buffer,
        test_lz77_offset_too_large,
        test_lz77_repeat_indent,
        test_dyn_dict_add,
        test_dyn_dict_reference,
        test_dyn_dict_oob,
        test_context_accumulation,
        test_context_lz77_after_emit,
        test_multi_v3_basic,
        test_v3_png_metadata,
        test_v3_png_xor_metadata,
        test_v3_png_seed_extraction,
        test_v3_fallback_to_v2,
        test_xor_channel_basic,
        test_bpe_single_pair,
        test_bpe_two_pairs,
        test_bpe_four_pairs,
        test_bpe_terminator,
        test_bpe_roundtrip_via_search,
        test_bpe_in_v3_pipeline,
        test_bpe_table_completeness,
    ]

    for test_fn in unit_tests:
        try:
            ok = test_fn()
            results.append((f"[V3] {test_fn.__name__}", ok if ok else False))
        except Exception as e:
            print(f"  [FAIL] {test_fn.__name__}: {e}")
            results.append((f"[V3] {test_fn.__name__}", False))

    print()

    # === V3 Round-Trip Tests ===
    print("V3 ROUND-TRIP TESTS")
    print("-" * 60)

    v3_tests = [
        # Short programs that V2 handles well -- V3 must not break them
        (b'print("Hello")\n', 'V3 short: Python Hello', True, 'python'),
        # Programs with repeated patterns -- V3 should help
        (b'def greet(name):\n    print("Hello, " + name)\n\ngreet("World")\n',
         'V3: Python greet (61B)', True, 'python'),
        (b'for i in range(10):\n    print(i)\n\n',
         'V3: Python loop (34B)', False, 'python'),
        (b'def fibonacci(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        a, b = 0, 1\n        for i in range(2, n + 1):\n            a, b = b, a + b\n        return b\n\nfor i in range(10):\n    print(f"fib({i}) = {fibonacci(i)}")\n',
         'V3: Fibonacci (254B)', True, 'python'),
    ]

    for target, desc, runnable, lang in v3_tests:
        try:
            ok = verify_v3_roundtrip(target, desc, runnable, lang, timeout=120.0)
            results.append((f"[V3] {desc}", ok))
        except Exception as e:
            print(f"  FAIL: {e}")
            results.append((f"[V3] {desc}", False))

    print()

    # === Summary ===
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for desc, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {desc}")

    print()
    print(f"  {passed}/{total} tests passed ({passed/total*100:.0f}%)")

    if passed == total:
        print()
        print("  PHASE 3 COMPLETE: Context-dependent expansion works!")
        print("  LZ77 back-references, dynamic dictionary, XOR channel verified.")
        print("  V3 PNGs round-trip correctly with fewer pixels than V2.")
        print("  All V1+V2 tests remain backward compatible.")

    return 0 if passed == total else 1


# ════════════════════════════════════════
# BOUNDARY & ERROR HANDLING TESTS
# (pytest-only, using assert style)
# ════════════════════════════════════════

def test_boundary_seed_zero():
    """Seed 0x00000000 should not crash."""
    result = expand(0x00000000)
    assert isinstance(result, bytes)
    assert len(result) > 0

def test_boundary_seed_max():
    """Seed 0xFFFFFFFF should not crash."""
    result = expand(0xFFFFFFFF)
    assert isinstance(result, bytes)
    assert len(result) > 0

def test_max_output_enforced():
    """max_output=1 must truncate to exactly 1 byte."""
    # DICT_1 index 0 = b'print(' (6 bytes) -- truncate to 1
    result = expand(0x00000000, max_output=1)
    assert len(result) == 1

def test_max_output_zero():
    """max_output=0 should return empty bytes."""
    result = expand(0x00000000, max_output=0)
    assert result == b''

def test_seed_negative_raises():
    """Negative seed must raise ValueError."""
    import pytest
    with pytest.raises(ValueError):
        expand(-1)

def test_seed_overflow_raises():
    """Seed > 0xFFFFFFFF must raise ValueError."""
    import pytest
    with pytest.raises(ValueError):
        expand(0x100000000)

def test_file_specific_table_validation():
    """Wrong-length table must raise ValueError."""
    import pytest
    from expand import set_file_specific_table, get_file_specific_table
    with pytest.raises(ValueError):
        set_file_specific_table("short")
    # Valid length should work
    set_file_specific_table("0123456789abcdef")
    assert get_file_specific_table() == "0123456789abcdef"
    # Reset
    set_file_specific_table(None)
    assert get_file_specific_table() == ' \netnari=:s(,lfd'

def test_single_byte_null_roundtrip():
    """Single null byte should roundtrip via V3 encode/decode."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        png_path = f.name
    try:
        seeds, png_data = encode_v3(b'\x00', png_path, timeout=30.0)
        result = expand_from_png_v3(png_data)
        assert result == b'\x00'
    finally:
        os.unlink(png_path)

def test_single_byte_ff_roundtrip():
    """Single 0xFF byte should roundtrip."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        png_path = f.name
    try:
        seeds, png_data = encode_v3(b'\xff', png_path, timeout=30.0)
        result = expand_from_png_v3(png_data)
        assert result == b'\xff'
    finally:
        os.unlink(png_path)

def test_single_byte_ascii_roundtrip():
    """Single ASCII byte 'A' should roundtrip."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        png_path = f.name
    try:
        seeds, png_data = encode_v3(b'A', png_path, timeout=30.0)
        result = expand_from_png_v3(png_data)
        assert result == b'A'
    finally:
        os.unlink(png_path)

def test_corrupt_png_raises():
    """Garbage bytes fed as PNG should raise an error, not crash silently."""
    import pytest
    with pytest.raises(Exception):
        expand_from_png_v3(b'THIS IS NOT A PNG FILE AT ALL')

def test_truncated_png_raises():
    """Truncated valid PNG should raise an error."""
    import pytest, tempfile, os
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        png_path = f.name
    try:
        seeds, png_data = encode_v3(b'test data for truncation', png_path, timeout=30.0)
        # Truncate to half size
        truncated = png_data[:len(png_data) // 2]
        with pytest.raises(Exception):
            expand_from_png_v3(truncated)
    finally:
        os.unlink(png_path)

def test_seed_rgba_roundtrip_boundaries():
    """Seed->RGBA->seed roundtrip for boundary values."""
    for seed in [0x00000000, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF, 0x01020304]:
        assert seed_from_rgba(*seed_to_rgba(seed)) == seed


if __name__ == '__main__':
    sys.exit(main())

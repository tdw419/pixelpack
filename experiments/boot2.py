"""
Pixelpack Phase 2 - Multi-Pixel PNG Encoder/Decoder

Extends boot.py with:
  - make_multipixel_png(): NxM RGBA PNG with N seeds (+ padding in square image)
  - read_multipixel_png(): extract real seeds from a multi-pixel PNG
  - encode_multi(): encode a program into a multi-pixel PNG
  - decode_png(): decode a multi-pixel PNG back to bytes
  - DP-based segmentation for minimum pixel count

Backward compatible: 1x1 PNGs (no tEXt chunk) work exactly as V1.
Multi-pixel PNGs include a tEXt chunk with seed count to separate real
seeds from padding pixels.
"""

import struct
import zlib
import math
import time
from expand import (
    seed_to_rgba, seed_from_rgba, expand, 
    DICTIONARY, DICTIONARY_EXT, SUB_DICT, NIBBLE_TABLE,
    BPE_PAIR_TABLE, _expand_nibble
)
from expand2 import expand_multi, expand_from_png, extract_seeds_from_png
from find_seed import search, _decompose, _pack_dict_seed, _verify


# ============================================================
# PNG Construction
# ============================================================

def make_multipixel_png(seeds: list) -> bytes:
    """
    Create an NxM RGBA PNG containing the given seeds as pixel colors.
    
    Uses square-ish dimensions for visual appeal. Adds a tEXt chunk
    with seed count so the decoder knows real vs padding pixels.
    """
    n = len(seeds)
    width, height = _auto_dimensions(n)
    
    # Build raw pixel data with filter byte 0 (none) per row
    raw_rows = bytearray()
    for row in range(height):
        raw_rows.append(0)  # filter byte = none
        for col in range(width):
            idx = row * width + col
            if idx < n:
                r, g, b, a = seed_to_rgba(seeds[idx])
            else:
                # Padding: black transparent (won't be decoded as real seed)
                r, g, b, a = 0, 0, 0, 0
            raw_rows.extend([r, g, b, a])
    
    compressed = zlib.compress(bytes(raw_rows))
    return _build_png(width, height, compressed, n)


def read_multipixel_png(png_data: bytes) -> tuple:
    """
    Read a multi-pixel PNG.
    
    Returns:
        (width, height, seeds) where seeds is ONLY the real seeds
        (padding excluded via tEXt chunk metadata).
    """
    all_seeds, real_count, _tables = extract_seeds_from_png(png_data)
    
    # Get dimensions from IHDR
    width = height = 0
    pos = 8
    while pos < len(png_data):
        if pos + 8 > len(png_data):
            break
        length = struct.unpack('>I', png_data[pos:pos+4])[0]
        chunk_type = png_data[pos+4:pos+8]
        data = png_data[pos+8:pos+8+length]
        if chunk_type == b'IHDR':
            width, height = struct.unpack('>II', data[:8])
            break
        pos += 12 + length
    
    return width, height, all_seeds[:real_count]


def encode_multi(target: bytes, output_png: str, timeout: float = None,
                 max_seeds: int = 0):
    """
    Encode a target byte sequence into a multi-pixel PNG.
    
    Strategy:
    1. Try to find a single seed first (1x1 PNG)
    2. If that fails, use DP segmentation to find minimum-pixel encoding
    3. Write the multi-pixel PNG with seed count metadata
    
    timeout: seconds. None or 0 = auto-scale (1s per 1KB of target).
    max_seeds: 0 = unlimited.
    """
    start_time = time.time()
    
    # Auto-scale timeout: 2 seconds per 1KB of target, minimum 10s, max 600s
    if not timeout:
        timeout = min(600.0, max(10.0, len(target) / 1024.0 * 2))
    
    print(f"Encoding: {len(target)} bytes (timeout={timeout:.0f}s)")
    try:
        print(f"  Text: {target.decode('ascii')!r}")
    except UnicodeDecodeError:
        print(f"  Hex: {target.hex()[:80]}...")
    print()
    
    # Step 1: Try single seed
    results = search(target, timeout=min(timeout, 10.0))
    if results:
        seed = results[0][0]
        png_data = make_multipixel_png([seed])
        with open(output_png, 'wb') as f:
            f.write(png_data)
        print(f"Encoded as 1x1 PNG ({len(png_data)} bytes)")
        return True
    
    # Step 2: Multi-seed encoding via DP
    print("Single seed not found. Trying multi-pixel encoding...")
    remaining_time = timeout - (time.time() - start_time)
    seeds = _find_multi_seeds_dp(target, remaining_time, max_seeds)
    
    if not seeds:
        print("FAILED: Could not encode target")
        return False
    
    png_data = make_multipixel_png(seeds)
    with open(output_png, 'wb') as f:
        f.write(png_data)
    
    width, height = _auto_dimensions(len(seeds))
    print(f"Encoded as {width}x{height} PNG ({len(seeds)} seeds, {len(png_data)} bytes)")
    
    # Verify via actual PNG round-trip
    decoded = expand_from_png(png_data)
    if decoded == target:
        print("Verification: PASS")
        return True
    else:
        print("Verification: FAIL")
        print(f"  Expected {len(target)} bytes, got {len(decoded)} bytes")
        # Show first diff
        for i in range(min(len(decoded), len(target))):
            if decoded[i] != target[i]:
                print(f"  First diff at byte {i}: expected 0x{target[i]:02X}, got 0x{decoded[i]:02X}")
                break
        return False


def decode_png(png_path: str, output_path: str = None):
    """Decode a multi-pixel PNG back to bytes."""
    with open(png_path, 'rb') as f:
        png_data = f.read()
    
    width, height, seeds = read_multipixel_png(png_data)
    
    print(f"Decoding: {png_path}")
    print(f"  Dimensions: {width}x{height}")
    print(f"  Seeds: {len(seeds)}")
    for i, s in enumerate(seeds):
        print(f"    [{i}] 0x{s:08X}")
    
    result = expand_multi(seeds)
    print(f"  Output: {len(result)} bytes")
    try:
        print(f"  Text: {result.decode('ascii')!r}")
    except UnicodeDecodeError:
        print(f"  Hex: {result.hex()}")
    
    if output_path:
        with open(output_path, 'wb') as f:
            f.write(result)
        import os
        os.chmod(output_path, 0o755)
        print(f"  Written to: {output_path}")
    
    return result


# ============================================================
# DP Segmentation Engine
# ============================================================

def _find_multi_seeds_dp(target: bytes, timeout: float, max_seeds: int) -> list:
    """
    Find minimum-pixel encoding using dynamic programming.
    
    Phase 1: Build coverage table of all (pos, length) -> seed matches
    Phase 2: DP to find shortest path from pos 0 to len(target)
    Phase 3: Extract optimal segmentation
    Phase 4: Fill any remaining gaps with search() fallback
    """
    start_time = time.time()
    tlen = len(target)
    
    # Phase 1: Build comprehensive coverage table
    # matches[pos] = list of (length, seed, strategy_name) 
    # We want ALL valid matches at each position, not just the longest
    matches = [[] for _ in range(tlen)]
    
    for pos in range(tlen):
        remaining = tlen - pos
        if time.time() - start_time > timeout * 0.7:
            break
        
        suffix = target[pos:]
        
        # --- DICT_N (0x0-0x6): base dictionary, 1-7 entries ---
        # Try all n values, collect all valid matches
        for n in range(1, 8):
            decomp = _try_prefix_decompose(suffix, n, DICTIONARY)
            if decomp:
                dlen = sum(len(DICTIONARY[i]) for i in decomp)
                seed = _pack_dict_seed(n, decomp)
                if _verify(seed, target[pos:pos+dlen]):
                    matches[pos].append((dlen, seed, f"DICT_{n}"))
        
        # --- DICTX5 (0x8): 5 entries from DICTIONARY_EXT (5-bit indices) ---
        decomp = _try_prefix_decompose(suffix, 5, DICTIONARY_EXT)
        if decomp and all(i < 32 for i in decomp):
            dlen = sum(len(DICTIONARY_EXT[i]) for i in decomp)
            params = sum((idx & 0x1F) << (5 * i) for i, idx in enumerate(decomp))
            seed = 0x80000000 | params
            if _verify(seed, target[pos:pos+dlen]):
                matches[pos].append((dlen, seed, "DICTX5"))
        
        # --- BPE (0x9): 4 x 7-bit byte-pair indices ---
        pair_to_idx_bpe = {}
        for _bi, _bp in enumerate(BPE_PAIR_TABLE):
            if _bi > 0 and _bp:
                pair_to_idx_bpe[_bp] = _bi
        for n_pairs_bpe in range(min(4, remaining // 2), 0, -1):
            pair_len_bpe = n_pairs_bpe * 2
            if pair_len_bpe > remaining:
                continue
            indices_bpe = []
            valid_bpe = True
            for pi in range(n_pairs_bpe):
                pair_bpe = target[pos + pi*2 : pos + pi*2 + 2]
                idx_bpe = pair_to_idx_bpe.get(pair_bpe)
                if idx_bpe is None:
                    valid_bpe = False
                    break
                indices_bpe.append(idx_bpe)
            if not valid_bpe:
                continue
            params_bpe = 0
            for i in range(4):
                if i < n_pairs_bpe:
                    params_bpe |= (indices_bpe[i] & 0x7F) << (7 * i)
            seed_bpe = 0x90000000 | params_bpe
            if _verify(seed_bpe, target[pos:pos+pair_len_bpe]):
                matches[pos].append((pair_len_bpe, seed_bpe, "BPE"))
        
        # --- DICTX7 (0xA): 7 entries from SUB_DICT (4-bit indices) ---
        decomp = _try_prefix_decompose(suffix, 7, SUB_DICT)
        if decomp:
            dlen = sum(len(SUB_DICT[i]) for i in decomp)
            params = sum((idx & 0xF) << (4 * i) for i, idx in enumerate(decomp))
            seed = 0xA0000000 | params
            if _verify(seed, target[pos:pos+dlen]):
                matches[pos].append((dlen, seed, "DICTX7"))
        
        # --- NIBBLE (0x7): exactly 7 bytes from NIBBLE_TABLE ---
        if remaining >= 7:
            nibble_match = _try_nibble(suffix[:7])
            if nibble_match is not None:
                matches[pos].append((7, nibble_match, "NIBBLE"))
        
        # --- BYTEPACK (0xE): 3-5 byte segments ---
        for seg_len in range(min(5, remaining), 2, -1):
            seg = target[pos:pos + seg_len]
            seed = _quick_bytepack(seg)
            if seed is not None:
                matches[pos].append((seg_len, seed, "BYTEPACK"))
        
        # Sort by length descending (prefer longer matches)
        matches[pos].sort(key=lambda x: -x[0])
    
    # Phase 2: DP for minimum-pixel path
    INF = float('inf')
    dp = [INF] * (tlen + 1)
    dp[tlen] = 0
    parent = [None] * (tlen + 1)
    
    for pos in range(tlen - 1, -1, -1):
        for length, seed, name in matches[pos]:
            end = pos + length
            if end <= tlen and dp[end] + 1 < dp[pos]:
                dp[pos] = dp[end] + 1
                parent[pos] = (length, seed, name)
    
    # Phase 3: Check if DP found a complete path
    effective_max = max_seeds if max_seeds > 0 else (tlen + 1)  # unlimited
    if dp[0] <= effective_max and parent[0] is not None:
        seeds = []
        pos = 0
        while pos < tlen:
            if parent[pos] is None:
                break
            length, seed, name = parent[pos]
            seeds.append(seed)
            print(f"  Segment {len(seeds)}: {length}B @ offset {pos} -> 0x{seed:08X} ({name})")
            pos += length
        
        if pos == tlen:
            return seeds
    
    # Phase 4: Fill gaps with search() fallback
    return _fill_gaps(target, matches, dp, parent, timeout, max_seeds, start_time, tlen)


def _fill_gaps(target, matches, dp, parent, timeout, max_seeds, start_time, tlen):
    """
    Fill uncovered positions using full search() as fallback.
    
    Walk through the target, using DP matches where available,
    calling search() for gap positions.
    """
    seeds = []
    pos = 0
    effective_max = max_seeds if max_seeds > 0 else (tlen + 1)
    
    while pos < tlen and len(seeds) < effective_max:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            print(f"  Timeout after {len(seeds)} segments")
            return []
        
        # Check if DP found a match here
        if parent[pos] is not None:
            length, seed, name = parent[pos]
            seeds.append(seed)
            print(f"  Segment {len(seeds)}: {length}B @ offset {pos} -> 0x{seed:08X} ({name})")
            pos += length
            continue
        
        # No DP match - try search() for decreasing segment lengths
        remaining = tlen - pos
        per_seg_timeout = min(0.5, (timeout - elapsed) / max(remaining, 1))
        found = False
        
        for seg_len in range(min(20, remaining), 0, -1):
            if time.time() - start_time > timeout:
                break
            segment = target[pos:pos + seg_len]
            results = search(segment, timeout=per_seg_timeout)
            if results:
                seeds.append(results[0][0])
                print(f"  Segment {len(seeds)}: {seg_len}B @ offset {pos} -> 0x{results[0][0]:08X} ({results[0][1]})")
                pos += seg_len
                found = True
                break
        
        if not found:
            print(f"  Cannot encode byte at offset {pos}: 0x{target[pos]:02X}")
            return []
    
    if pos != tlen:
        print(f"  Only encoded {pos}/{tlen} bytes")
        return []
    
    return seeds


# ============================================================
# Decomposition Helpers
# ============================================================

def _try_prefix_decompose(target, n_entries, dictionary):
    """
    Decompose a PREFIX of target into exactly n_entries dict entries.
    Returns list of indices or None.
    """
    return _prefix_decomp_rec(target, 0, n_entries, dictionary)


def _prefix_decomp_rec(target, pos, remaining, dictionary):
    if remaining == 0:
        return []  # all entries matched, prefix is target[:pos]
    if pos >= len(target):
        return None
    
    for i, entry in enumerate(dictionary):
        elen = len(entry)
        if pos + elen <= len(target) and target[pos:pos + elen] == entry:
            rest = _prefix_decomp_rec(target, pos + elen, remaining - 1, dictionary)
            if rest is not None:
                return [i] + rest
    return None


def _try_nibble(segment):
    """Try to encode a 7-byte segment via NIBBLE strategy. Returns seed or None."""
    if len(segment) != 7:
        return None
    byte_to_nibble = {}
    for i, b in enumerate(NIBBLE_TABLE):
        byte_to_nibble[b] = i
    nibbles = []
    for b in segment:
        if b not in byte_to_nibble:
            return None
        nibbles.append(byte_to_nibble[b])
    params = 0
    for i, nib in enumerate(nibbles):
        params |= (nib & 0xF) << (4 * i)
    seed = 0x70000000 | params
    if _verify(seed, segment):
        return seed
    return None


def _quick_bytepack(target):
    """
    Fast BYTEPACK check without calling full search().
    Checks multiple modes for 3-5 byte segments.
    Returns seed or None.
    """
    tlen = len(target)
    
    # Mode 6: 5 bytes via Python-source table (top 32 chars by frequency)
    # Uses file-specific table when set, otherwise falls back to default.
    if tlen == 5:
        try:
            from expand import get_file_specific_mode6_table
            table = get_file_specific_mode6_table()
        except ImportError:
            table = ' etab\nr\'sni,d)(lxop=y0u_:Fc-fm1"'
        try:
            indices = [table.index(chr(b)) for b in target]
            data = sum(idx << (5 * i) for i, idx in enumerate(indices))
            seed = 0xE0000000 | (6 << 0) | (data << 3)
            if _verify(seed, target):
                return seed
        except (ValueError, OverflowError):
            pass
    
    # Mode 7: 5 bytes via extended Python-source table (chars ranked 33-64)
    if tlen == 5:
        table = 'I>2#ETg&hAC.B43675[]DP8+NvLRk\\\\XS'
        try:
            indices = [table.index(chr(b)) for b in target]
            data = sum(idx << (5 * i) for i, idx in enumerate(indices))
            seed = 0xE0000000 | (7 << 0) | (data << 3)
            if _verify(seed, target):
                return seed
        except (ValueError, OverflowError):
            pass
    
    # Mode 1: 4 bytes via 6-bit indices into 64-char table + optional repeats of last byte
    # Uses file-specific table when set, otherwise falls back to default.
    # Decoder supports 4 + len_flag bytes (len_flag 0-7), so we can encode 4-11 bytes.
    if 4 <= tlen <= 11:
        try:
            from expand import get_file_specific_mode1_table
            table = get_file_specific_mode1_table()
        except ImportError:
            table = (
                ' etaoinsrlhdcu.mfpgwybvk\n"\',)(\']-=_:;<>{}[]!@#'
                '0123456789/\\+*%&|?^~`$'
            )[:64]
        try:
            indices = [table.index(chr(b)) for b in target[:4]]
            len_flag = tlen - 4
            can_repeat = len_flag == 0 or all(
                target[4 + i] == target[3] for i in range(len_flag)
            )
            if can_repeat:
                data = sum(idx << (6 * i) for i, idx in enumerate(indices))
                if len_flag > 0:
                    data |= (len_flag << 24)
                seed = 0xE0000000 | (1 << 0) | (data << 3)
                if _verify(seed, target):
                    return seed
        except (ValueError, OverflowError):
            pass
    
    # Mode 3: Compact 6-char via 4-bit indices into Python-source table
    # Uses file-specific table when set, otherwise falls back to default.
    if tlen == 6:
        try:
            from expand import get_file_specific_table
            table = get_file_specific_table()
        except ImportError:
            table = ' \netnari=:s(,lfd'
        try:
            indices = [table.index(chr(b)) for b in target]
            data = sum(idx << (4 * i) for i, idx in enumerate(indices))
            seed = 0xE0000000 | (3 << 0) | (data << 3)
            if _verify(seed, target):
                return seed
        except (ValueError, OverflowError):
            pass
    
    # Mode 0: 3 raw bytes + optional repeat of first byte
    if tlen >= 3:
        b0, b1, b2 = target[0], target[1], target[2]
        extra = tlen - 3
        if extra <= 15:
            if extra == 0 or all(target[3 + i] == b0 for i in range(extra)):
                data = b0 | (b1 << 8) | (b2 << 16) | (extra << 24)
                seed = 0xE0000000 | (0 << 0) | (data << 3)
                if _verify(seed, target):
                    return seed
    
    # Mode 2: ADD delta (3-4 bytes)
    if tlen == 3:
        base = target[0]
        d1 = (target[1] - base) & 0xFF
        d2 = (target[2] - target[1]) & 0xFF
        data = base | (d1 << 8) | (d2 << 16)
        seed = 0xE0000000 | (2 << 0) | (data << 3)
        if _verify(seed, target):
            return seed
    elif tlen == 4:
        base = target[0]
        d1 = (target[1] - base) & 0xFF
        d2 = (target[2] - target[1]) & 0xFF
        d3 = (target[3] - target[2]) & 0xF
        data = base | (d1 << 8) | (d2 << 16) | (d3 << 24)
        seed = 0xE0000000 | (2 << 0) | (data << 3)
        if _verify(seed, target):
            return seed
    
    # Mode 4: 4 bytes, 7 bits each
    if tlen == 4 and all(b <= 127 for b in target):
        data = (target[0] & 0x7F) | ((target[1] & 0x7F) << 7) | \
               ((target[2] & 0x7F) << 14) | ((target[3] & 0x7F) << 21)
        seed = 0xE0000000 | (4 << 0) | (data << 3)
        if _verify(seed, target):
            return seed
    
    # Mode 5: Shared base + 4 nibble offsets
    if tlen == 4:
        for base in range(256):
            offsets = [(b - base) & 0xFF for b in target]
            if all(0 <= o <= 15 for o in offsets):
                data = base | (offsets[0] << 8) | (offsets[1] << 12) | \
                       (offsets[2] << 16) | (offsets[3] << 20)
                seed = 0xE0000000 | (5 << 0) | (data << 3)
                if _verify(seed, target):
                    return seed
    
    return None


# ============================================================
# PNG Helpers
# ============================================================

def _auto_dimensions(n):
    """Choose image dimensions for n seeds."""
    if n <= 0:
        return 1, 1
    if n == 1:
        return 1, 1
    if n == 2:
        return 2, 1
    # Find smallest square that fits n
    side = math.ceil(math.sqrt(n))
    return side, side


def _build_png(width, height, compressed_data, seed_count=None):
    """Build a PNG file from dimensions and compressed IDAT data."""
    def chunk(chunk_type, data):
        c = chunk_type + data
        crc = zlib.crc32(c) & 0xFFFFFFFF
        return struct.pack('>I', len(data)) + c + struct.pack('>I', crc)
    
    signature = b'\x89PNG\r\n\x1a\n'
    ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0)
    ihdr = chunk(b'IHDR', ihdr_data)
    
    # Add tEXt chunk with seed count (so decoder knows real vs padding)
    chunks = [signature, ihdr]
    if seed_count is not None and seed_count > 0:
        text_data = b'seedcnt\x00' + str(seed_count).encode('ascii')
        chunks.append(chunk(b'tEXt', text_data))
    
    idat = chunk(b'IDAT', compressed_data)
    iend = chunk(b'IEND', b'')
    chunks.extend([idat, iend])
    
    return b''.join(chunks)


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Pixelpack Phase 2 - Multi-Pixel PNG Encoder/Decoder")
        print()
        print("Usage:")
        print("  python boot2.py encode <input_file> <output.png>")
        print("  python boot2.py decode <input.png> [output_file]")
        print("  python boot2.py demo")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == 'encode':
        if len(sys.argv) < 4:
            print("Usage: python boot2.py encode <input_file> <output.png>")
            sys.exit(1)
        success = encode_multi(
            open(sys.argv[2], 'rb').read(),
            sys.argv[3],
            timeout=0  # auto-scale
        )
        sys.exit(0 if success else 1)
    
    elif cmd == 'decode':
        if len(sys.argv) < 3:
            print("Usage: python boot2.py decode <input.png> [output_file]")
            sys.exit(1)
        output = sys.argv[3] if len(sys.argv) > 3 else None
        result = decode_png(sys.argv[2], output)
        sys.exit(0)
    
    elif cmd == 'demo':
        print("=" * 60)
        print("PIXELPACK PHASE 2 DEMO - Multi-Pixel Encoding")
        print("=" * 60)
        print()
        
        demos = [
            (b'print("Hello")\n', 'V1 backward compat (1x1)'),
            (b'echo Hello\n', 'Shell echo (1x1)'),
            (b'x = "Hello"\nprint(x)\n', 'Python variable (multi-pixel)'),
        ]
        
        for target, desc in demos:
            print(f"--- {desc} ---")
            print(f"Target: {target!r} ({len(target)} bytes)")
            
            png_path = f'/tmp/pixelpack_demo_{hash(target) % 10000}.png'
            success = encode_multi(target, png_path, timeout=15.0)
            
            if success:
                decoded = decode_png(png_path)
                if decoded == target:
                    print("Round-trip: PASS")
                else:
                    print("Round-trip: FAIL")
                    print(f"  Expected: {target.hex()}")
                    print(f"  Got:      {decoded.hex()}")
            else:
                print("Encoding: FAILED")
            print()
    
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

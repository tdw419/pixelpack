"""
Pixelpack Phase 2 - Multi-Pixel Expansion

Extends expand.py with:
  - expand_multi(): chain multiple seeds into one output
  - expand_from_png(): decode a multi-pixel PNG to bytes
  - extract_seeds_from_png(): read seeds from PNG pixels

Backward compatible: all V1 seeds still expand identically via expand_v1.
The scaling mechanism is multi-pixel chaining: N seeds = N independent
expansions concatenated.
"""

import struct
import zlib
from expand import expand as expand_v1, seed_from_rgba


def expand_multi(seeds: list, max_output: int = 10_000_000) -> bytes:
    """
    Expand multiple seeds into one concatenated byte sequence.
    
    Each seed expands independently, results concatenate left-to-right.
    This is the core of multi-pixel encoding.
    """
    result = bytearray()
    for seed in seeds:
        if len(result) >= max_output:
            break
        expanded = expand_v1(seed, max_output - len(result))
        result.extend(expanded)
    return bytes(result)


def expand_from_png(png_data: bytes) -> bytes:
    """
    Expand a PNG (1x1 or multi-pixel) into bytes.
    
    Reads the seed count from the tEXt chunk (if present) to know
    how many pixels are real seeds vs padding. Also extracts any
    file-specific tables (bp8table, bp_mode6_table, bp_mode1_table)
    and sets them before expanding seeds.
    """
    seeds, real_count, tables = extract_seeds_from_png(png_data)
    
    # Set file-specific tables before expanding
    from expand import (set_file_specific_table, set_file_specific_mode6_table,
                        set_file_specific_mode1_table,
                        set_freq_table, set_keyword_table)
    if tables.get('bp8table'):
        set_file_specific_table(tables['bp8table'])
    if tables.get('bp_mode6_table'):
        set_file_specific_mode6_table(tables['bp_mode6_table'])
    if tables.get('bp_mode1_table'):
        set_file_specific_mode1_table(tables['bp_mode1_table'])
    if tables.get('freq_table'):
        set_freq_table(tables['freq_table'])
    if tables.get('keyword_table'):
        set_keyword_table(tables['keyword_table'])
    
    try:
        result = expand_multi(seeds[:real_count])
    finally:
        # Always clear tables after use
        set_file_specific_table(None)
        set_file_specific_mode6_table(None)
        set_file_specific_mode1_table(None)
        set_freq_table(None)
        set_keyword_table(None)
    
    return result


def extract_seeds_from_png(png_data: bytes) -> tuple:
    """
    Extract seeds from a multi-pixel PNG.
    
    Returns:
        (seeds_list, real_count, tables) where real_count is the number of
        actual seeds (from tEXt chunk), rest are padding, and tables is a
        dict of file-specific tables extracted from tEXt chunks.
        If no tEXt chunk, real_count = len(seeds_list) (V1 compat).
    """
    if png_data[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError("Not a valid PNG file")
    
    pos = 8
    width = height = 0
    idat_data = b''
    real_count = None
    tables = {}
    
    while pos < len(png_data):
        if pos + 8 > len(png_data):
            break
        length = struct.unpack('>I', png_data[pos:pos+4])[0]
        chunk_type = png_data[pos+4:pos+8]
        data = png_data[pos+8:pos+8+length]
        
        if chunk_type == b'IHDR':
            width, height = struct.unpack('>II', data[:8])
        elif chunk_type == b'IDAT':
            idat_data += data
        elif chunk_type == b'tEXt':
            # tEXt format: keyword\0value
            null_pos = data.find(b'\x00')
            if null_pos >= 0:
                keyword = data[:null_pos].decode('ascii', errors='ignore')
                value = data[null_pos+1:]
                if keyword == 'seedcnt':
                    real_count = int(value.decode('ascii', errors='ignore'))
                elif keyword in ('bp8table', 'bp_mode6_table', 'bp_mode1_table'):
                    # Table values are hex-encoded latin-1 strings
                    tables[keyword] = bytes.fromhex(value.decode('ascii', errors='ignore')).decode('latin-1')
                elif keyword == 'freq_table':
                    # 256-byte frequency-ranked table, hex-encoded
                    tables['freq_table'] = bytes.fromhex(value.decode('ascii', errors='ignore'))
                elif keyword == 'keyword_table':
                    # Keywords separated by 0xFF, hex-encoded
                    kw_raw = bytes.fromhex(value.decode('ascii', errors='ignore'))
                    tables['keyword_table'] = [kw for kw in kw_raw.split(b'\xff') if kw]
        
        pos += 12 + length
    
    if not idat_data:
        raise ValueError("No IDAT chunk found")
    
    decompressed = zlib.decompress(idat_data)
    
    # Parse pixel data (filter byte per row)
    bpp = 4  # RGBA
    stride = 1 + width * bpp
    seeds = []
    
    for row in range(height):
        row_start = row * stride
        if row_start >= len(decompressed):
            break
        filter_byte = decompressed[row_start]
        if filter_byte != 0:
            _apply_filter(decompressed, row_start, width, bpp, filter_byte, height, stride)
        
        for col in range(width):
            px = row_start + 1 + col * bpp
            if px + 4 > len(decompressed):
                break
            r, g, b, a = decompressed[px:px+4]
            seeds.append(seed_from_rgba(r, g, b, a))
    
    # If no tEXt chunk, all pixels are real (V1 compat)
    if real_count is None:
        real_count = len(seeds)
    
    return seeds, real_count, tables


def _apply_filter(data, row_start, width, bpp, filter_type, height, stride):
    """Apply PNG row filter to reconstruct raw pixel data (in-place)."""
    if filter_type == 0:
        return
    
    row_len = width * bpp
    for i in range(row_len):
        pos = row_start + 1 + i
        if pos >= len(data):
            break
        x = data[pos]
        
        a = data[pos - bpp] if i >= bpp else 0
        b = data[pos - stride] if row_start > 0 else 0
        c = data[pos - stride - bpp] if (row_start > 0 and i >= bpp) else 0
        
        if filter_type == 1:    # Sub
            data[pos] = (x + a) & 0xFF
        elif filter_type == 2:  # Up
            data[pos] = (x + b) & 0xFF
        elif filter_type == 3:  # Average
            data[pos] = (x + ((a + b) >> 1)) & 0xFF
        elif filter_type == 4:  # Paeth
            data[pos] = (x + _paeth_predictor(a, b, c)) & 0xFF


def _paeth_predictor(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    elif pb <= pc:
        return b
    else:
        return c


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Pixelpack Phase 2 - Multi-Pixel Expansion")
        print()
        print("Usage:")
        print("  python expand2.py <seed_hex> [<seed_hex2> ...]")
        print("  python expand2.py --png <file.png>")
        sys.exit(1)
    
    if sys.argv[1] == '--png':
        with open(sys.argv[2], 'rb') as f:
            png_data = f.read()
        result = expand_from_png(png_data)
        seeds, count, _tables = extract_seeds_from_png(png_data)
        print(f"Seeds: {count} real / {len(seeds)} total")
        for i, s in enumerate(seeds[:count]):
            print(f"  [{i}] 0x{s:08X}")
    else:
        seeds = [int(s, 16) for s in sys.argv[1:]]
        result = expand_multi(seeds)
    
    print(f"Output: {len(result)} bytes")
    print(f"Hex: {result.hex()}")
    try:
        print(f"ASCII: {result.decode('ascii')!r}")
    except UnicodeDecodeError:
        print(f"Raw: {result!r}")

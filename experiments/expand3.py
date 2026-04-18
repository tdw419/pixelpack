"""
Pixelpack Phase 3 - Context-Dependent Expansion

Extends expand.py/expand2.py with strategies that reference previously
expanded output. This enables:
  - LZ77 back-references for repeated substrings
  - Dynamic dictionary for repeated tokens
  - XOR channel for inter-pixel delta encoding

The key architectural change: expansion is no longer purely functional.
An ExpandContext object accumulates state across pixel expansions.

Backward compatibility: V1/V2 PNGs (no p3mode tEXt chunk) decode
identically using expand_v1. Only PNGs with t3mode=1 use context.

Strategy mapping in phase 3 mode:
  0x0-0xB: Same as V1 (delegated to expand_v1)
  0xC: LZ77_BACKREF - copy from output buffer
  0xD: DYN_DICT - dynamic dictionary add/reference
  0xE: CONTEXT_BYTEPACK - enhanced bytepack that can use V1's BYTEPACK
       or reference dyn_dict entries
  0xF: TEMPLATE (same as V1, plus adds to dyn_dict)

Actually, to keep it simpler and more robust:
  0x0-0xB: Delegate to expand_v1 (unchanged)
  0xC: LZ77_BACKREF
  0xD: DYN_DICT
  0xE: V1 BYTEPACK (unchanged, but output added to dyn_dict tracking)
  0xF: V1 TEMPLATE (unchanged, but output added to dyn_dict tracking)
"""

from dataclasses import dataclass, field
from expand import expand as expand_v1, seed_from_rgba, seed_to_rgba
from expand2 import expand_multi, expand_from_png, extract_seeds_from_png


@dataclass
class ExpandContext:
    """State that persists across pixel expansions within one decode."""
    output_buffer: bytearray = field(default_factory=bytearray)
    dyn_dict: list = field(default_factory=list)  # list of bytes objects
    prev_seed: int = 0  # previous raw seed value (for XOR channel)
    xor_mode: bool = False  # if True, XOR each seed with previous


def expand_with_context(seed: int, ctx: ExpandContext) -> bytes:
    """
    Expand a seed using context-dependent strategies.
    
    For strategies 0x0-0xB, delegates to expand_v1 (no context needed).
    For strategies 0xC-0xF, uses the ExpandContext for back-references.
    """
    # Apply XOR channel if enabled
    if ctx.xor_mode and len(ctx.output_buffer) > 0:
        seed = seed ^ ctx.prev_seed
    ctx.prev_seed = seed  # track for next XOR

    strategy = (seed >> 28) & 0xF
    params = seed & 0x0FFFFFFF

    if strategy <= 0xA:
        # V1 strategies 0x0-0xA -- no context needed, unchanged behavior
        result = expand_v1(seed)
    elif strategy == 0xB:
        # FREQ_TABLE: frequency-ranked byte encoding (V3 mode override)
        from expand import expand_freq_table, get_freq_table
        if get_freq_table() is not None:
            result = expand_freq_table(params)
        else:
            result = expand_v1(seed)  # fallback to RLE
    elif strategy == 0xC:
        result = _expand_lz77(params, ctx)
    elif strategy == 0xD:
        # KEYWORD_TABLE: keyword lookup encoding (V3 mode override)
        from expand import expand_keyword_table, get_keyword_table
        if get_keyword_table() is not None:
            result = expand_keyword_table(params)
        else:
            result = _expand_dyn_dict(params, ctx)  # fallback to DYN_DICT
    elif strategy == 0xE:
        # In phase 3 mode, 0xE is still BYTEPACK (V1 behavior)
        result = expand_v1(seed)
    elif strategy == 0xF:
        # TEMPLATE -- V1 behavior
        result = expand_v1(seed)
    else:
        result = b''

    # Append to output buffer for LZ77 referencing
    ctx.output_buffer.extend(result)
    return result


def emit_dict_seed(seed: int, ctx: ExpandContext) -> bytes:
    """
    Emit a seed's expansion into the reference buffer ONLY (not output).
    Used for setup pixels that establish LZ77 targets without
    appearing in the final program output.
    """
    result = expand_v1(seed)
    ctx.output_buffer.extend(result)
    return result


def expand_multi_v3(seeds: list, max_output: int = 65536) -> bytes:
    """
    Expand multiple seeds with context-dependent strategies.
    
    Creates a fresh ExpandContext and processes seeds sequentially.
    This is the V3 equivalent of expand2.expand_multi().
    """
    ctx = ExpandContext()
    result = bytearray()
    for seed in seeds:
        if len(result) >= max_output:
            break
        expanded = expand_with_context(seed, ctx)
        result.extend(expanded)
    return bytes(result[:max_output])


def expand_multi_v3_xor(seeds: list, max_output: int = 65536) -> bytes:
    """Like expand_multi_v3 but with XOR channel enabled."""
    ctx = ExpandContext(xor_mode=True)
    result = bytearray()
    for seed in seeds:
        if len(result) >= max_output:
            break
        expanded = expand_with_context(seed, ctx)
        result.extend(expanded)
    return bytes(result[:max_output])


def expand_from_png_v3(png_data: bytes) -> bytes:
    """
    Expand a PNG using context-dependent strategies.
    
    Reads the t3mode flag from tEXt chunk. If present and =1,
    uses context-dependent expansion. Otherwise falls back to V2.
    
    Seeds before dict_only count are "setup seeds": they populate
    the LZ77 reference buffer but don't appear in output.
    
    If a 'bp8table' tEXt chunk is present, sets the file-specific
    BYTEPACK table before expanding seeds, and resets it after.
    """
    p3mode = _read_p3mode(png_data)
    xor_mode = _read_xor_mode(png_data)
    
    if not p3mode:
        # Not a phase 3 PNG -- use V2 expansion
        return expand_from_png(png_data)
    
    seeds, real_count, _tables = extract_seeds_from_png(png_data)
    real_seeds = seeds[:real_count]
    dict_only = _read_dict_only_count(png_data)
    
    # Handle file-specific BYTEPACK table
    bp8table_hex = _read_text_chunk(png_data, 'bp8table')
    if bp8table_hex:
        from expand import set_file_specific_table
        try:
            table_str = bytes.fromhex(bp8table_hex).decode('latin-1')
            set_file_specific_table(table_str)
        except (ValueError, UnicodeDecodeError):
            pass
    
    # Handle file-specific mode-6 BYTEPACK table
    bp_mode6_hex = _read_text_chunk(png_data, 'bp_mode6_table')
    if bp_mode6_hex:
        from expand import set_file_specific_mode6_table
        try:
            table_str = bytes.fromhex(bp_mode6_hex).decode('latin-1')
            set_file_specific_mode6_table(table_str)
        except (ValueError, UnicodeDecodeError):
            pass
    
    # Handle file-specific mode-1 BYTEPACK table (64-char)
    bp_mode1_hex = _read_text_chunk(png_data, 'bp_mode1_table')
    if bp_mode1_hex:
        from expand import set_file_specific_mode1_table
        try:
            table_str = bytes.fromhex(bp_mode1_hex).decode('latin-1')
            set_file_specific_mode1_table(table_str)
        except (ValueError, UnicodeDecodeError):
            pass
    
    # Handle frequency-ranked byte table (for FREQ_TABLE strategy 0xB)
    freq_table_hex = _read_text_chunk(png_data, 'freq_table')
    if freq_table_hex:
        from expand import set_freq_table
        try:
            freq_bytes = bytes.fromhex(freq_table_hex)
            set_freq_table(freq_bytes)
        except (ValueError, UnicodeDecodeError):
            pass
    
    # Handle keyword table (for KEYWORD_TABLE strategy 0xD)
    keyword_table_hex = _read_text_chunk(png_data, 'keyword_table')
    if keyword_table_hex:
        from expand import set_keyword_table
        try:
            kw_data = bytes.fromhex(keyword_table_hex)
            # Format: N bytes of keywords separated by 0xFF sentinel
            # Each keyword is a sequence of non-0xFF bytes
            keywords = []
            current = bytearray()
            for b in kw_data:
                if b == 0xFF:
                    if current:
                        keywords.append(bytes(current))
                        current = bytearray()
                else:
                    current.append(b)
            if current:
                keywords.append(bytes(current))
            set_keyword_table(keywords if keywords else None)
        except (ValueError, UnicodeDecodeError):
            pass
    
    # Handle BPE pair table (for BPE strategy 0x9)
    bpe_table_hex = _read_text_chunk(png_data, 'bpe_table')
    if bpe_table_hex:
        from expand import set_file_specific_bpe_table
        try:
            bpe_data = bytes.fromhex(bpe_table_hex)
            # Format: pairs of bytes (2 bytes each), fill table indices 1..127
            custom_bpe = [b'']  # index 0 = terminator
            for i in range(0, len(bpe_data) - 1, 2):
                custom_bpe.append(bpe_data[i:i+2])
            while len(custom_bpe) < 128:
                custom_bpe.append(b'')
            set_file_specific_bpe_table(custom_bpe)
        except (ValueError, UnicodeDecodeError):
            pass
    
    try:
        ctx = ExpandContext(xor_mode=xor_mode)
        result = bytearray()
        
        for i, seed in enumerate(real_seeds):
            if i < dict_only:
                # Dict-only seed: populate reference buffer, no output
                emit_dict_seed(seed, ctx)
            else:
                expanded = expand_with_context(seed, ctx)
                result.extend(expanded)
        
        return bytes(result)
    finally:
        # Always reset tables to default after expansion
        if bp8table_hex:
            from expand import set_file_specific_table
            set_file_specific_table(None)
        if bp_mode6_hex:
            from expand import set_file_specific_mode6_table
            set_file_specific_mode6_table(None)
        if bp_mode1_hex:
            from expand import set_file_specific_mode1_table
            set_file_specific_mode1_table(None)
        if freq_table_hex:
            from expand import set_freq_table
            set_freq_table(None)
        if keyword_table_hex:
            from expand import set_keyword_table
            set_keyword_table(None)
        if bpe_table_hex:
            from expand import set_file_specific_bpe_table
            set_file_specific_bpe_table(None)


# ============================================================
# LZ77 Back-Reference Strategy (0xC)
# ============================================================

def _expand_lz77(params: int, ctx: ExpandContext) -> bytes:
    """
    LZ77 back-reference into the accumulated output buffer.
    
    Param layout (28 bits):
      [15:0]  offset  (16 bits) - distance back from end of buffer
      [27:16] length  (12 bits) - number of bytes to copy
    
    Copy `length` bytes starting at position 
    `(len(output_buffer) - 1 - offset)`.
    
    If offset >= buffer length, returns empty bytes.
    Handles overlapping copies (length > offset) by copying byte-by-byte.
    """
    offset = params & 0xFFFF
    length = (params >> 16) & 0xFFF
    
    buf_len = len(ctx.output_buffer)
    if buf_len == 0 or offset >= buf_len:
        return b''
    
    # Start position: go back (offset+1) from end
    start = buf_len - 1 - offset
    if start < 0:
        start = 0
    
    # Copy byte-by-byte to handle overlapping refs (like LZ77)
    result = bytearray()
    for i in range(length):
        pos = start + i
        if pos < len(ctx.output_buffer):
            result.append(ctx.output_buffer[pos])
        elif pos < len(ctx.output_buffer) + len(result):
            # Overlapping: reference already-copied bytes in this result
            result.append(result[pos - len(ctx.output_buffer)])
        else:
            break
    
    return bytes(result)


# ============================================================
# Dynamic Dictionary Strategy (0xD)
# ============================================================

def _expand_dyn_dict(params: int, ctx: ExpandContext) -> bytes:
    """
    Dynamic dictionary: add or reference entries.
    
    Two modes selected by bit 27:
    
    Reference mode (bit 27 = 0):
      [26:0] index into dynamic dictionary
      Looks up the bytes at that index and emits them.
      
    Add mode (bit 27 = 1):
      [23:0] encode a literal string using base dictionary lookups
      (4 bits per entry, up to 6 entries from DICTIONARY)
      The expanded bytes get added to the dynamic dictionary
      AND emitted as output.
    """
    from expand import DICTIONARY
    
    is_add = (params >> 27) & 1
    
    if is_add:
        # Add mode: expand using base dictionary, add to dyn_dict
        # 24 bits = 6 x 4-bit indices into base DICTIONARY
        n_entries = ((params >> 24) & 0x7)  # bits 26:24 = count (1-7)
        if n_entries == 0:
            n_entries = 1
        entry_indices = []
        for i in range(n_entries):
            idx = (params >> (4 * i)) & 0xF
            entry_indices.append(idx)
        
        result = bytearray()
        for idx in entry_indices:
            result.extend(DICTIONARY[idx])
        result = bytes(result)
        
        # Add to dynamic dictionary
        if result:
            ctx.dyn_dict.append(result)
        
        return result
    else:
        # Reference mode: look up by index
        index = params & 0x7FFFFFFF  # 27-bit index
        if index < len(ctx.dyn_dict):
            return ctx.dyn_dict[index]
        return b''


# ============================================================
# PNG Metadata Helpers
# ============================================================

def _read_p3mode(png_data: bytes) -> bool:
    """Check if PNG has tEXt chunk with t3mode=1."""
    return _read_text_chunk(png_data, 't3mode') == '1'


def _read_xor_mode(png_data: bytes) -> bool:
    """Check if PNG has tEXt chunk with xor_mode=true."""
    return _read_text_chunk(png_data, 'xor_mode') == 'true'


def _read_dict_only_count(png_data: bytes) -> int:
    """Read the number of dict-only setup seeds."""
    val = _read_text_chunk(png_data, 'dict_only')
    return int(val) if val else 0


def _read_text_chunk(png_data: bytes, keyword: str) -> str:
    """Read a tEXt chunk value by keyword."""
    import struct
    if png_data[:8] != b'\x89PNG\r\n\x1a\n':
        return ''
    pos = 8
    while pos < len(png_data):
        if pos + 8 > len(png_data):
            break
        length = struct.unpack('>I', png_data[pos:pos+4])[0]
        chunk_type = png_data[pos+4:pos+8]
        data = png_data[pos+8:pos+8+length]
        if chunk_type == b'tEXt':
            null_pos = data.find(b'\x00')
            if null_pos >= 0:
                key = data[:null_pos].decode('ascii', errors='ignore')
                if key == keyword:
                    return data[null_pos+1:].decode('ascii', errors='ignore')
        pos += 12 + length
    return ''


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Pixelpack Phase 3 - Context-Dependent Expansion")
        print()
        print("Usage:")
        print("  python3 expand3.py <seed_hex> [<seed_hex2> ...]")
        print("  python3 expand3.py --png <file.png>")
        sys.exit(1)
    
    if sys.argv[1] == '--png':
        with open(sys.argv[2], 'rb') as f:
            png_data = f.read()
        result = expand_from_png_v3(png_data)
        print(f"Output: {len(result)} bytes")
        try:
            print(f"ASCII: {result.decode('ascii')!r}")
        except UnicodeDecodeError:
            print(f"Hex: {result.hex()}")
    else:
        seeds = [int(s, 16) for s in sys.argv[1:]]
        result = expand_multi_v3(seeds)
        print(f"Output: {len(result)} bytes")
        try:
            print(f"ASCII: {result.decode('ascii')!r}")
        except UnicodeDecodeError:
            print(f"Hex: {result.hex()}")

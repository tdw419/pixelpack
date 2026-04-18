"""
Pixelpack Phase 3 - Context-Aware Encoder v3

Two-phase approach:
  1. SETUP: Pre-emit high-value repeated patterns into the reference buffer
     (dict_only seeds). These don't appear in output but establish LZ77 targets.
  2. ENCODE: Encode the target using LZ77 back-references (to setup + emitted
     content) and V1 strategies for unique content.

Key insight: Only pre-emit patterns that cost MORE in V1 than setup+LZ77.
A pattern needing N V1 pixels that appears K times costs N*K pixels in V2.
With setup, it costs setup_px + K LZ77 pixels. Savings = N*K - setup_px - K.

Produces V3 PNGs with t3mode=1 tEXt chunk and dict_only=N metadata.
"""

import struct
import zlib
import math
import time
from expand import (
    seed_to_rgba,
    DICTIONARY, DICTIONARY_EXT, SUB_DICT, BPE_PAIR_TABLE,
)
from expand2 import expand_multi, extract_seeds_from_png
from expand3 import (
    ExpandContext, expand_with_context, expand_from_png_v3,
    _expand_lz77, emit_dict_seed,
)
from find_seed import search as seed_search
from boot2 import (
    _find_multi_seeds_dp, make_multipixel_png,
    _try_prefix_decompose, _try_nibble, _quick_bytepack,
)
from collections import Counter


# ============================================================
# File-Specific BYTEPACK Table
# ============================================================

def _build_optimal_bytepack_table(target: bytes, min_improvement: float = 0.05) -> str:
    """Build an optimal 16-char BYTEPACK table for the given target bytes.
    
    Returns the table as a 16-char string, or None if the default table
    is already good enough (improvement < min_improvement).
    
    The table is ranked by byte frequency: index 0 = most frequent byte.
    """
    DEFAULT_TABLE = ' \netnari=:s(,lfd'
    
    freq = Counter(target)
    total = len(target)
    
    # Compute default table coverage
    default_covered = sum(freq.get(ord(c), 0) for c in DEFAULT_TABLE)
    default_pct = default_covered / total if total > 0 else 0
    
    # Build optimal table: top 16 bytes by frequency
    optimal_chars = [chr(b) for b, _ in freq.most_common(16)]
    # Pad to 16 if fewer unique bytes
    while len(optimal_chars) < 16:
        optimal_chars.append('\x00')
    optimal_table = ''.join(optimal_chars)
    
    # Compute optimal table coverage
    optimal_covered = sum(freq.get(ord(c), 0) for c in optimal_table)
    optimal_pct = optimal_covered / total if total > 0 else 0
    
    improvement = optimal_pct - default_pct
    if improvement < min_improvement:
        return None
    
    return optimal_table


def _build_optimal_mode6_table(target: bytes, min_improvement: float = 0.05) -> str:
    """Build an optimal 32-char BYTEPACK mode-6 table for the given target bytes.
    
    Returns the table as a 32-char string, or None if the default table
    is already good enough (improvement < min_improvement).
    
    Improvement is measured by 5-byte run coverage (the actual metric that
    determines how many mode-6 seeds the DP encoder can find), not individual
    byte coverage.
    """
    DEFAULT_TABLE = ' etab\nr\'sni,d)(lxop=y0u_:Fc-fm1"'
    
    freq = Counter(target)
    total = len(target)
    
    # Build optimal table: top 32 bytes by frequency
    optimal_chars = [chr(b) for b, _ in freq.most_common(32)]
    # Pad to 32 if fewer unique bytes
    while len(optimal_chars) < 32:
        optimal_chars.append('\x00')
    optimal_table = ''.join(optimal_chars)
    
    # Measure improvement by 5-byte run coverage (what actually matters for mode-6)
    def _count_5byte_runs(data, table_str):
        table_set = set(ord(c) for c in table_str)
        count = 0
        for i in range(len(data) - 4):
            if all(data[j] in table_set for j in range(i, i+5)):
                count += 1
        return count
    
    default_runs = _count_5byte_runs(target, DEFAULT_TABLE)
    optimal_runs = _count_5byte_runs(target, optimal_table)
    
    # Use absolute run count improvement; default min_improvement=0.05 means
    # optimal needs at least 5% more runs (relative to total possible positions)
    total_positions = max(total - 4, 1)
    improvement = (optimal_runs - default_runs) / total_positions
    
    if improvement < min_improvement:
        return None
    
    return optimal_table


def _build_optimal_mode1_table(target: bytes, min_improvement: float = 0.02) -> str:
    """Build an optimal 64-char BYTEPACK mode-1 table for the given target bytes.
    
    Returns the table as a 64-char string, or None if the default table
    is already good enough (improvement < min_improvement).
    
    Mode-1 encodes 4 bytes per seed using 6-bit indices into a 64-char table.
    Improvement is measured by 4-byte run coverage.
    """
    # Default table from expand.py - we reconstruct it here to avoid circular import
    DEFAULT_TABLE = (
        ' etaoinsrlhdcu.mfpgwybvk\n"x,)(\']-=_:;<>{}[]!@#'
        '0123456789/\\+*%&|?^~`$'
    )[:64]
    
    freq = Counter(target)
    total = len(target)
    
    # Build optimal table: top 64 bytes by frequency
    optimal_chars = [chr(b) for b, _ in freq.most_common(64)]
    while len(optimal_chars) < 64:
        optimal_chars.append('\x00')
    optimal_table = ''.join(optimal_chars)
    
    # Measure improvement by 4-byte run coverage (what actually matters for mode-1)
    def _count_4byte_runs(data, table_str):
        table_set = set(ord(c) for c in table_str)
        count = 0
        for i in range(len(data) - 3):
            if all(data[j] in table_set for j in range(i, i+4)):
                count += 1
        return count
    
    default_runs = _count_4byte_runs(target, DEFAULT_TABLE)
    optimal_runs = _count_4byte_runs(target, optimal_table)
    
    total_positions = max(total - 3, 1)
    improvement = (optimal_runs - default_runs) / total_positions
    
    if improvement < min_improvement:
        return None
    
    return optimal_table

def make_v3_png(seeds: list, xor_mode: bool = False, dict_only: int = 0,
                bp8table: str = None, bp_mode6_table: str = None,
                bp_mode1_table: str = None,
                freq_table: bytes = None, keyword_table: list = None,
                bpe_table: list = None) -> bytes:
    """Create a PNG with phase 3 metadata. dict_only = number of setup seeds.
    
    bp8table: optional 16-char string for file-specific BYTEPACK mode 3 table.
    bp_mode6_table: optional 32-char string for file-specific BYTEPACK mode 6 table.
    bp_mode1_table: optional 64-char string for file-specific BYTEPACK mode 1 table.
    freq_table: optional 256-byte frequency-ranked table for FREQ_TABLE strategy.
    keyword_table: optional list of bytes objects for KEYWORD_TABLE strategy.
    bpe_table: optional list of 128 byte-pair entries for custom BPE table.
    """
    n = len(seeds)
    width, height = _auto_dimensions(n)
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
    return _build_v3_png(width, height, compressed, n, xor_mode, dict_only,
                         bp8table, bp_mode6_table, bp_mode1_table,
                         freq_table, keyword_table, bpe_table)


def _build_v3_png(width, height, compressed_data, seed_count, xor_mode=False,
                   dict_only=0, bp8table=None, bp_mode6_table=None, bp_mode1_table=None,
                   freq_table=None, keyword_table=None, bpe_table=None):
    def chunk(chunk_type, data):
        c = chunk_type + data
        crc = zlib.crc32(c) & 0xFFFFFFFF
        return struct.pack('>I', len(data)) + c + struct.pack('>I', crc)

    signature = b'\x89PNG\r\n\x1a\n'
    ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0)
    ihdr = chunk(b'IHDR', ihdr_data)
    chunks = [signature, ihdr]
    chunks.append(chunk(b'tEXt', b'seedcnt\x00' + str(seed_count).encode()))
    chunks.append(chunk(b'tEXt', b't3mode\x001'))
    if dict_only > 0:
        chunks.append(chunk(b'tEXt', b'dict_only\x00' + str(dict_only).encode()))
    if xor_mode:
        chunks.append(chunk(b'tEXt', b'xor_mode\x00true'))
    if bp8table is not None:
        table_hex = bp8table.encode('latin-1').hex()
        chunks.append(chunk(b'tEXt', b'bp8table\x00' + table_hex.encode()))
    if bp_mode6_table is not None:
        table_hex = bp_mode6_table.encode('latin-1').hex()
        chunks.append(chunk(b'tEXt', b'bp_mode6_table\x00' + table_hex.encode()))
    if bp_mode1_table is not None:
        table_hex = bp_mode1_table.encode('latin-1').hex()
        chunks.append(chunk(b'tEXt', b'bp_mode1_table\x00' + table_hex.encode()))
    if freq_table is not None:
        chunks.append(chunk(b'tEXt', b'freq_table\x00' + freq_table.hex().encode()))
    if keyword_table is not None:
        # Serialize: keywords joined by 0xFF sentinel
        kw_bytes = b'\xff'.join(keyword_table)
        chunks.append(chunk(b'tEXt', b'keyword_table\x00' + kw_bytes.hex().encode()))
    if bpe_table is not None:
        # Serialize: concat all 127 pairs (254 bytes), index 0 omitted
        bpe_bytes = b''.join(p for p in bpe_table[1:] if p and len(p) == 2)
        chunks.append(chunk(b'tEXt', b'bpe_table\x00' + bpe_bytes.hex().encode()))
    idat = chunk(b'IDAT', compressed_data)
    iend = chunk(b'IEND', b'')
    chunks.extend([idat, iend])
    return b''.join(chunks)


def _auto_dimensions(n):
    if n <= 0: return 1, 1
    if n == 1: return 1, 1
    if n == 2: return 2, 1
    side = math.ceil(math.sqrt(n))
    return side, side


# ============================================================
# LZ77 Helpers
# ============================================================

def _make_lz77_seed(offset, length):
    """Create an LZ77 seed. Returns None if params don't fit."""
    if offset >= (1 << 16) or length >= (1 << 12) or length < 1:
        return None
    params = offset | (length << 16)
    seed = 0xC0000000 | params
    return seed


def _verify_lz77(offset, length, emitted, expected):
    """Verify that LZ77 produces the expected bytes."""
    ctx = ExpandContext()
    ctx.output_buffer = bytearray(emitted)
    params = offset | (length << 16)
    result = _expand_lz77(params, ctx)
    return result == expected


def _find_lz77_at(target, pos, emitted):
    """Find longest LZ77 match for target[pos:] in emitted buffer."""
    buf_len = len(emitted)
    if buf_len == 0:
        return None
    remaining = len(target) - pos
    max_len = min(remaining, 0xFFF)

    best_len = 0
    best_offset = 0

    # For each possible start position in the buffer
    for start in range(buf_len):
        match_len = 0
        ei = start
        while match_len < max_len:
            if ei < buf_len:
                if emitted[ei] == target[pos + match_len]:
                    match_len += 1
                    ei += 1
                else:
                    break
            else:
                # Overlapping copy: reference bytes we're about to produce
                wrap_pos = ei - buf_len
                if wrap_pos < match_len and (pos + wrap_pos) < len(target):
                    if target[pos + wrap_pos] == target[pos + match_len]:
                        match_len += 1
                        ei += 1
                    else:
                        break
                else:
                    break

        if match_len > best_len:
            best_len = match_len
            best_offset = buf_len - 1 - start

    if best_len >= 2 and best_offset < (1 << 16):
        if _verify_lz77(best_offset, best_len, emitted, target[pos:pos+best_len]):
            return best_len, best_offset
    return None



# ============================================================
# BPE Helpers
# ============================================================

# Cached reverse lookup for BPE pair -> index
_bpe_pair_to_idx_cache = None
_bpe_pair_to_idx_table_id = None  # track which table the cache was built from

def _bpe_pair_to_idx():
    """Build and cache reverse lookup: byte pair -> BPE table index.
    Uses file-specific table when set, otherwise default BPE_PAIR_TABLE.
    Cache is invalidated when the table changes.
    """
    global _bpe_pair_to_idx_cache, _bpe_pair_to_idx_table_id
    from expand import get_file_specific_bpe_table
    current_table = get_file_specific_bpe_table()
    table_id = id(current_table)
    if _bpe_pair_to_idx_cache is None or _bpe_pair_to_idx_table_id != table_id:
        _bpe_pair_to_idx_cache = {}
        for i, pair in enumerate(current_table):
            if i > 0 and pair:
                _bpe_pair_to_idx_cache[pair] = i
        _bpe_pair_to_idx_table_id = table_id
    return _bpe_pair_to_idx_cache


def _try_bpe(suffix):
    """Try BPE encoding for suffix. Returns (length, seed, "BPE") or None.

    Tries to cover 2-8 bytes with 1-4 byte-pair lookups.
    Returns the longest valid match.
    """
    remaining = len(suffix)
    pair_to_idx = _bpe_pair_to_idx()

    # Find the longest BPE match (greedy on pairs from start)
    for n_pairs in range(min(4, remaining // 2), 0, -1):
        pair_len = n_pairs * 2
        indices = []
        valid = True
        for pi in range(n_pairs):
            pair = suffix[pi*2 : pi*2 + 2]
            idx = pair_to_idx.get(pair)
            if idx is None:
                valid = False
                break
            indices.append(idx)
        if not valid:
            continue

        # Pack 4 x 7-bit indices (unused = 0 = terminator)
        params = 0
        for i in range(4):
            if i < n_pairs:
                params |= (indices[i] & 0x7F) << (7 * i)

        seed = 0x90000000 | params
        from find_seed import _verify
        if _verify(seed, suffix[:pair_len]):
            return (pair_len, seed, "BPE")

    return None


# ============================================================
# V1 Match Finding
# ============================================================

def _find_v1_match(target, pos):
    """Find the best V1 strategy match at target[pos:]."""
    remaining = len(target) - pos
    suffix = target[pos:]

    best = (0, None, "")

    # DICT_N (1-7)
    for n in range(1, 8):
        decomp = _try_prefix_decompose(suffix, n, DICTIONARY)
        if decomp:
            dlen = sum(len(DICTIONARY[i]) for i in decomp)
            if dlen > best[0]:
                from find_seed import _pack_dict_seed
                seed = _pack_dict_seed(n, decomp)
                from find_seed import _verify
                if _verify(seed, target[pos:pos+dlen]):
                    best = (dlen, seed, f"DICT_{n}")

    # DICTX5
    decomp = _try_prefix_decompose(suffix, 5, DICTIONARY_EXT)
    if decomp and all(i < 32 for i in decomp):
        dlen = sum(len(DICTIONARY_EXT[i]) for i in decomp)
        params = sum((idx & 0x1F) << (5 * i) for i, idx in enumerate(decomp))
        seed = 0x80000000 | params
        from find_seed import _verify
        if dlen > best[0] and _verify(seed, target[pos:pos+dlen]):
            best = (dlen, seed, "DICTX5")

    # BPE (0x9): byte-pair encoding
    bpe_match = _try_bpe(suffix)
    if bpe_match and bpe_match[0] > best[0]:
        best = bpe_match

    # DICTX7
    decomp = _try_prefix_decompose(suffix, 7, SUB_DICT)
    if decomp:
        dlen = sum(len(SUB_DICT[i]) for i in decomp)
        params = sum((idx & 0xF) << (4 * i) for i, idx in enumerate(decomp))
        seed = 0xA0000000 | params
        from find_seed import _verify
        if dlen > best[0] and _verify(seed, target[pos:pos+dlen]):
            best = (dlen, seed, "DICTX7")

    # NIBBLE
    if remaining >= 7:
        nib = _try_nibble(suffix[:7])
        if nib and 7 > best[0]:
            best = (7, nib, "NIBBLE")

    # BYTEPACK
    max_bp = min(18, remaining)
    for seg_len in range(max_bp, 2, -1):
        if seg_len > 5 and seg_len > 3 and target[pos + 3] != target[pos]:
            continue
        seg = target[pos:pos + seg_len]
        seed = _quick_bytepack(seg)
        if seed and seg_len > best[0]:
            best = (seg_len, seed, "BYTEPACK")

    return best if best[0] > 0 else None


# ============================================================
# Setup Pattern Analysis
# ============================================================

def _find_setup_candidates(target, max_setup_seeds=50, time_budget=12.0):
    """
    Find repeated substrings worth pre-emitting into the LZ77 reference buffer.
    
    Scans for repeated substrings of length 3-50 that appear at least twice.
    Evaluates each using actual DP encoding cost, computing net savings:
      savings = v1_cost_per_occurrence * occurrences - setup_cost - occurrences
    
    Args:
        target: Bytes to analyze for repeated patterns
        max_setup_seeds: Maximum number of setup seeds to allocate
        time_budget: Seconds before giving up on analysis
    
    Returns:
        List of (pattern, v1_seeds, occurrence_count, net_savings) tuples,
        sorted by savings descending, with non-overlapping positions.
    """
    t_start = time.time()
    tlen = len(target)

    # Fast path: skip setup entirely for very large files
    if tlen > 50000:
        return []

    # Adjust pattern length range for larger files
    max_pattern = 50 if tlen <= 4000 else (40 if tlen <= 15000 else (25 if tlen <= 30000 else 14))
    min_pattern = 3

    # Phase 1: Find repeated substrings using hash-based grouping.
    # For efficiency, group by 4-byte prefix hash first, then deduplicate.
    candidates = {}

    for length in range(max_pattern, min_pattern - 1, -1):
        if time.time() - t_start > time_budget * 0.2:
            break
        seen = {}
        for i in range(tlen - length + 1):
            sub = target[i:i+length]
            if sub in seen:
                if sub not in candidates:
                    candidates[sub] = [seen[sub]]
                if i not in candidates[sub]:
                    candidates[sub].append(i)
            else:
                seen[sub] = i

    # Filter to patterns appearing >= 2 times
    candidates = {k: v for k, v in candidates.items() if len(v) >= 2}

    # Phase 2: Pre-sort by estimated savings (frequency * length) to evaluate
    # the most promising candidates first. This lets us stop early when we've
    # found enough good patterns or hit the time budget.
    prelim = []
    for pattern, positions in candidates.items():
        occurrences = len(positions)
        # Estimated V1 cost: ~1 seed per 3-5 bytes (rough heuristic)
        est_v1_cost = max(1, (len(pattern) + 2) // 3)
        est_savings = est_v1_cost * occurrences - est_v1_cost - occurrences
        if est_savings > 0:
            prelim.append((est_savings, pattern, positions))

    prelim.sort(key=lambda x: -x[0])

    # Phase 3: Evaluate top candidates with actual DP, time-budgeted.
    # Stop after finding max_setup_seeds worth of patterns or running out of budget.
    dp_time_per = 0.5 if tlen <= 4000 else 0.4
    max_candidates = 40 if tlen <= 4000 else 25
    evaluated = 0

    scored = []
    for est_sav, pattern, positions in prelim:
        if evaluated >= max_candidates:
            break
        if time.time() - t_start > time_budget * 0.8:
            break

        occurrences = len(positions)

        # Quick check: if pattern fits in 1 V1 seed, LZ77 can't save
        # (1 seed either way). Skip the expensive DP call.
        if len(pattern) <= 4:
            # BYTEPACK covers up to 5 bytes in 1 seed
            continue

        v1_seeds = _find_multi_seeds_dp(pattern, timeout=dp_time_per, max_seeds=16)
        evaluated += 1
        if not v1_seeds:
            continue
        v1_cost = len(v1_seeds)

        if v1_cost <= 1:
            continue

        setup_cost = v1_cost
        net_savings = (v1_cost * occurrences) - (setup_cost + occurrences)
        if net_savings > 0:
            scored.append((pattern, v1_seeds, v1_cost, occurrences, positions, net_savings))

    scored.sort(key=lambda x: -x[5])

    # Phase 4: Select non-overlapping patterns greedily by savings
    selected = []
    covered_positions = set()

    for pattern, v1_seeds, v1_cost, occurrences, positions, net_savings in scored:
        usable_positions = []
        for pos in positions:
            overlap = False
            for j in range(len(pattern)):
                if pos + j in covered_positions:
                    overlap = True
                    break
            if not overlap:
                usable_positions.append(pos)

        if len(usable_positions) < 2:
            continue

        actual_savings = (v1_cost * len(usable_positions)) - (len(v1_seeds) + len(usable_positions))
        if actual_savings <= 0:
            continue

        for pos in usable_positions:
            for j in range(len(pattern)):
                covered_positions.add(pos + j)

        selected.append((pattern, v1_seeds, v1_cost, len(usable_positions), actual_savings))

    # Limit setup seeds
    total_setup = 0
    final = []
    for pattern, v1_seeds, v1_cost, count, savings in selected:
        if total_setup + len(v1_seeds) > max_setup_seeds:
            continue
        total_setup += len(v1_seeds)
        final.append((pattern, v1_seeds, count, savings))

    return final


# ============================================================
# V3 Encoder
# ============================================================

# ============================================================
# FREQ_TABLE and KEYWORD_TABLE
# ============================================================

def _build_freq_table(target: bytes) -> bytes:
    """Build a frequency-ranked byte table for FREQ_TABLE v4 strategy.
    
    Returns the top 38 bytes ranked by frequency (most common first).
    The first 7 entries use short-form encoding (0+3bit = 4 bits/byte).
    Entries 8-38 use long-form encoding (1+5bit = 6 bits/byte).
    Max 7 bytes per seed (all short-form), typical 4-7.
    """
    freq = Counter(target)
    ranked = sorted(freq.keys(), key=lambda b: (-freq[b], b))[:38]
    return bytes(ranked)


# Pre-built keyword tables for common file types
_PYTHON_KEYWORDS = [
    b'def ', b'class ', b'    ', b'return ', b'import ',
    b'from ', b'self.', b'= ', b'    ', b'\n    ',
    b'print(', b'.', b'if ', b'else:', b'elif ',
    b'for ', b'while ', b'in ', b'not ', b'and ',
    b'or ', b'is ', b'None', b'True', b'False',
    b'raise ', b'try:', b'except', b'with ', b'as ',
    b'pass', b'break', b'continue', b'yield ', b'lambda ',
    b'global ', b'nonlocal ', b'assert ', b'del ', b'finally:',
    b'"""', b"'''", b'# ', b'== ', b'!= ',
    b'>= ', b'<= ', b'+ ', b'- ', b'* ',
    b'/ ', b'% ', b'**', b'//', b'->',
    b': ', b', ', b'()\n', b'[]', b'{}',
]


def _build_keyword_table(target: bytes) -> list:
    """Build a keyword table from target bytes using file-specific n-gram analysis.
    
    Returns a list of up to 64 keyword byte strings, ranked by
    (frequency * length) -- the best compression candidates.
    
    Strategy: extract 3-6 byte n-grams from the file, rank by savings,
    then deduplicate (remove substrings of longer keywords).
    Also includes pre-built Python keywords as seeds.
    """
    tlen = len(target)
    if tlen < 10:
        return None
    
    # Collect n-gram frequencies for n=3..6
    ngram_freq = Counter()
    for n in range(6, 2, -1):  # 6, 5, 4, 3
        min_count = 3 if n <= 4 else 2
        for i in range(tlen - n + 1):
            ng = target[i:i+n]
            ngram_freq[ng] += 1
    
    # Score by savings: (freq * length) -- bytes covered if used as keyword
    scored = []
    for ng, count in ngram_freq.items():
        if count < 2:  # Must appear at least twice
            continue
        if len(ng) < 3:  # Minimum 3 bytes per keyword
            continue
        savings = count * len(ng)
        scored.append((savings, count, ng))
    
    # Also score pre-built Python keywords
    for kw in _PYTHON_KEYWORDS:
        if len(kw) < 3:
            continue
        count = 0
        pos = 0
        while True:
            idx = target.find(kw, pos)
            if idx == -1:
                break
            count += 1
            pos = idx + 1
        if count >= 2:
            savings = count * len(kw)
            scored.append((savings, count, kw))
    
    # Sort by savings descending
    scored.sort(key=lambda x: -x[0])
    
    # Deduplicate: remove n-grams that are substrings of longer, higher-ranked ones
    # Process in order of savings (highest first)
    keywords = []
    keyword_bytes = set()  # Track byte ranges covered to avoid overlap
    for savings, count, ng in scored:
        if len(keywords) >= 64:
            break
        # Skip if this n-gram is a substring of an already-selected keyword
        is_substring = False
        for existing in keywords:
            if ng in existing and len(ng) < len(existing):
                is_substring = True
                break
        if not is_substring:
            keywords.append(ng)
    
    return keywords if keywords else None


def _build_optimal_bpe_table(target: bytes, min_improvement: float = 0.10) -> list:
    """Build an optimal 127-entry BPE pair table for the given target bytes.
    
    Returns a list of 128 entries (index 0 = empty/terminator, 1-127 = byte pairs),
    or None if the default table is already good enough.
    
    Strategy: count all byte pairs, rank by frequency, pick top 127.
    Compare greedy coverage against default BPE_PAIR_TABLE.
    """
    from expand import BPE_PAIR_TABLE
    
    tlen = len(target)
    if tlen < 20:
        return None
    
    # Count byte pair frequencies
    pair_freq = Counter()
    for i in range(tlen - 1):
        pair_freq[target[i:i+2]] += 1
    
    # Build custom table: top 127 pairs
    top_pairs = [p for p, _ in pair_freq.most_common(127)]
    custom_table = [b''] + list(top_pairs)  # index 0 = terminator
    while len(custom_table) < 128:
        custom_table.append(b'')
    
    # Measure greedy coverage (non-overlapping) for both tables
    def greedy_coverage(data, table):
        pair_set = {}
        for i, pair in enumerate(table):
            if pair and len(pair) == 2:
                pair_set[pair] = i
        covered = 0
        i = 0
        while i < len(data) - 1:
            if data[i:i+2] in pair_set:
                covered += 2
                i += 2
            else:
                i += 1
        return covered
    
    default_covered = greedy_coverage(target, BPE_PAIR_TABLE)
    custom_covered = greedy_coverage(target, custom_table)
    
    improvement = (custom_covered - default_covered) / tlen
    if improvement < min_improvement:
        return None
    
    print(f"  Custom BPE table: {custom_covered}B vs default {default_covered}B "
          f"(+{custom_covered - default_covered}B = {improvement*100:.1f}% improvement)")
    return custom_table


def _try_freq_table_encode(segment: bytes, freq_table: bytes) -> tuple:
    """Try to encode a segment using FREQ_TABLE v4 strategy (0xB).
    
    Returns (seed, n_bytes_encoded) or None.
    
    V4 variable-width format:
      For each byte, encode as:
        - If in table[0..6]: 1-bit prefix (0) + 3-bit index (1-7), 4 bits total
        - If in table[7..37]: 1-bit prefix (1) + 5-bit index (1-31), 6 bits total
      Index 0 = terminator.
      Must fit in 28 bits total.
      Max 7 bytes (all short-form), typical 4-7.
    """
    if len(segment) < 1:
        return None
    
    # Build reverse lookup: byte_value -> (tier, index)
    # tier 0: table[0..6] -> short form (4 bits)
    # tier 1: table[7..37] -> long form (6 bits)
    lookup = {}
    for i in range(min(7, len(freq_table))):
        lookup[freq_table[i]] = (0, i + 1)  # short form, 1-indexed
    for i in range(7, min(38, len(freq_table))):
        lookup[freq_table[i]] = (1, i - 7 + 1)  # long form, 1-indexed
    
    # Encode bytes, tracking bit budget
    params = 0
    bit_pos = 0
    encoded_len = 0
    
    for b in segment:
        if b not in lookup:
            break  # Can't encode this byte
        tier, idx = lookup[b]
        if tier == 0:
            # Short form: 0 + 3-bit index
            if bit_pos + 4 > 28:
                break  # Out of bits
            params |= (0 << bit_pos)  # prefix 0
            params |= ((idx & 0x7) << (bit_pos + 1))
            bit_pos += 4
        else:
            # Long form: 1 + 5-bit index
            if bit_pos + 6 > 28:
                break  # Out of bits
            params |= (1 << bit_pos)  # prefix 1
            params |= ((idx & 0x1F) << (bit_pos + 1))
            bit_pos += 6
        encoded_len += 1
    
    if encoded_len == 0:
        return None
    
    seed = 0xB0000000 | params
    
    # Verify roundtrip
    from expand import expand_freq_table, set_freq_table, get_freq_table
    old_table = get_freq_table()
    set_freq_table(freq_table)
    try:
        result = expand_freq_table(params)
        if result == segment[:encoded_len]:
            return (seed, encoded_len)
    finally:
        set_freq_table(old_table)
    
    return None


def _try_keyword_table_encode(segment: bytes, keywords: list) -> tuple:
    """Try to encode a segment using KEYWORD_TABLE strategy (0xD).
    
    Returns (seed, n_bytes_encoded) or None.
    
    Tries to match the start of segment against keyword combinations.
    Bit layout: [3:0] count (1-4), [27:4] up to 4 x 6-bit indices.
    """
    if not keywords or len(segment) < 2:
        return None
    
    # Build keyword -> index map
    kw_to_idx = {}
    for i, kw in enumerate(keywords):
        if i >= 64:
            break
        kw_to_idx[kw] = i
    
    # Greedy: match as many keywords from the start as possible (max 4)
    indices = []
    pos = 0
    while len(indices) < 4 and pos < len(segment):
        matched = False
        # Try longest keyword first
        best_kw = None
        best_idx = -1
        for kw, idx in kw_to_idx.items():
            if segment[pos:pos+len(kw)] == kw and len(kw) > 0:
                if best_kw is None or len(kw) > len(best_kw):
                    best_kw = kw
                    best_idx = idx
        if best_kw is not None:
            indices.append(best_idx)
            pos += len(best_kw)
            matched = True
        if not matched:
            break
    
    if not indices:
        return None
    
    total_bytes = pos
    if total_bytes < 3:  # Minimum 3 bytes to be worth a keyword seed
        return None
    
    # Pack: count (4 bits) + up to 4 x 6-bit indices (24 bits)
    count = len(indices)
    data = 0
    for i, idx in enumerate(indices):
        data |= (idx & 0x3F) << (6 * i)
    params = count | (data << 4)
    seed = 0xD0000000 | params
    
    # Verify roundtrip
    from expand import expand_keyword_table, set_keyword_table, get_keyword_table
    old_kws = get_keyword_table()
    set_keyword_table(keywords)
    try:
        result = expand_keyword_table(params)
        if result == segment[:total_bytes]:
            return (seed, total_bytes)
    finally:
        set_keyword_table(old_kws)
    
    return None


def _try_keyword_hybrid_encode(segment: bytes, keywords: list) -> tuple:
    """Try hybrid KEYWORD_TABLE encoding: kw1 + literal_byte + kw2.
    
    Returns (seed, n_bytes_encoded) or None.
    
    Hybrid format (count=0):
      [3:0]   = 0 (hybrid marker)
      [9:4]   kw1 index (6 bits)
      [15:10] kw2 index (6 bits)
      [23:16] literal byte between kw1 and kw2
      [27:24] gap_size (0 = no gap/2 keywords, 1 = 1 literal byte)
    """
    if not keywords or len(segment) < 3:
        return None
    
    # Build keyword lookup (longest first for greedy matching)
    kw_by_len = sorted([(kw, i) for i, kw in enumerate(keywords) if i < 64], 
                        key=lambda x: -len(x[0]))
    
    best_seed = None
    best_bytes = 0
    
    # Try matching kw1 at the start
    for kw1, kw1_idx in kw_by_len:
        if len(kw1) < 2:
            continue
        if segment[:len(kw1)] != kw1:
            continue
        
        after_kw1 = len(kw1)
        
        # Try gap = 0 (just 2 keywords back-to-back)
        for kw2, kw2_idx in kw_by_len:
            if len(kw2) < 2:
                continue
            if segment[after_kw1:after_kw1+len(kw2)] != kw2:
                continue
            total = after_kw1 + len(kw2)
            if total > best_bytes and total >= 5:
                # Pack hybrid: count=0, kw1_idx, kw2_idx, literal=0, gap=0
                params = (kw1_idx << 4) | (kw2_idx << 10) | (0 << 16) | (0 << 24)
                seed = 0xD0000000 | params
                best_seed = seed
                best_bytes = total
            break  # longest kw2 match
        
        # Try gap = 1 (1 literal byte between keywords)
        if after_kw1 + 1 < len(segment):
            literal = segment[after_kw1]
            for kw2, kw2_idx in kw_by_len:
                if len(kw2) < 2:
                    continue
                start2 = after_kw1 + 1
                if segment[start2:start2+len(kw2)] != kw2:
                    continue
                total = start2 + len(kw2)
                if total > best_bytes and total >= 5:
                    # Pack hybrid: count=0, kw1_idx, kw2_idx, literal, gap=1
                    params = (kw1_idx << 4) | (kw2_idx << 10) | (literal << 16) | (1 << 24)
                    seed = 0xD0000000 | params
                    best_seed = seed
                    best_bytes = total
                break  # longest kw2 match
        
        break  # longest kw1 match (greedy)
    
    if best_seed is None or best_bytes < 5:
        return None
    
    # Verify roundtrip
    from expand import expand_keyword_table, set_keyword_table, get_keyword_table
    old_kws = get_keyword_table()
    set_keyword_table(keywords)
    try:
        result = expand_keyword_table(best_seed & 0x0FFFFFFF)
        if result == segment[:best_bytes]:
            return (best_seed, best_bytes)
    finally:
        set_keyword_table(old_kws)
    
    return None


def encode_v3(target: bytes, output_png: str = None, timeout: float = 120.0,
              use_xor: bool = False) -> tuple:
    """Encode target bytes into a V3 PNG with context-dependent strategies.
    
    Uses a three-phase approach:
      1. Build file-specific BYTEPACK table if beneficial
      2. Try encoding with and without LZ77 setup seeds
      3. Pick the encoding with fewest total pixels (never exceeds V2 baseline)
    
    Args:
        target: Raw bytes to encode
        output_png: If set, write PNG to this path
        timeout: Maximum encoding time in seconds
        use_xor: Enable XOR channel between seeds
    
    Returns:
        (data_seeds, png_data) on success, (None, None) on failure
    """
    start_time = time.time()
    tlen = len(target)

    print(f"V3 Encoding: {tlen} bytes")
    try:
        print(f"  Text: {target.decode('ascii')!r}")
    except UnicodeDecodeError:
        pass

    # Build file-specific BYTEPACK table if it helps
    bp8table = _build_optimal_bytepack_table(target, min_improvement=0.03)
    if bp8table:
        from expand import set_file_specific_table, get_file_specific_table
        set_file_specific_table(bp8table)
        freq = Counter(target)
        total = len(target)
        default_covered = sum(freq.get(ord(c), 0) for c in ' \netnari=:s(,lfd')
        bp8_covered = sum(freq.get(ord(c), 0) for c in bp8table)
        print(f"  File-specific BYTEPACK table: {bp8_covered}/{total} ({bp8_covered/total*100:.1f}%) vs default {default_covered}/{total} ({default_covered/total*100:.1f}%)")
    else:
        from expand import set_file_specific_table
        set_file_specific_table(None)

    # Build file-specific mode-6 table (32-char) if it helps
    bp_mode6_table = _build_optimal_mode6_table(target, min_improvement=0.03)
    if bp_mode6_table:
        from expand import set_file_specific_mode6_table, get_file_specific_mode6_table
        set_file_specific_mode6_table(bp_mode6_table)
        freq6 = Counter(target)
        total6 = len(target)
        default6_table = ' etab\nr\'sni,d)(lxop=y0u_:Fc-fm1"'
        default6_covered = sum(freq6.get(ord(c), 0) for c in default6_table)
        mode6_covered = sum(freq6.get(ord(c), 0) for c in bp_mode6_table)
        # Measure 5-byte run coverage (the real metric for mode-6)
        default6_set = set(ord(c) for c in default6_table)
        optimal6_set = set(ord(c) for c in bp_mode6_table)
        default6_runs = sum(1 for i in range(total6 - 4) if all(target[j] in default6_set for j in range(i, i+5)))
        optimal6_runs = sum(1 for i in range(total6 - 4) if all(target[j] in optimal6_set for j in range(i, i+5)))
        print(f"  File-specific mode-6 table: {optimal6_runs} 5-byte runs vs default {default6_runs} ({(optimal6_runs-default6_runs)/(total6-4)*100:.1f}% more coverage)")
    else:
        from expand import set_file_specific_mode6_table
        set_file_specific_mode6_table(None)

    # Build file-specific mode-1 table (64-char) if it helps
    bp_mode1_table = _build_optimal_mode1_table(target, min_improvement=0.02)
    if bp_mode1_table:
        from expand import set_file_specific_mode1_table
        set_file_specific_mode1_table(bp_mode1_table)
        freq1 = Counter(target)
        total1 = len(target)
        # Count 4-byte run coverage
        default1_table = (
            ' etaoinsrlhdcu.mfpgwybvk\n"x,)(\']-=_:;<>{}[]!@#'
            '0123456789/\\+*%&|?^~`$'
        )[:64]
        default1_set = set(ord(c) for c in default1_table)
        optimal1_set = set(ord(c) for c in bp_mode1_table)
        default1_runs = sum(1 for i in range(total1 - 3) if all(target[j] in default1_set for j in range(i, i+4)))
        optimal1_runs = sum(1 for i in range(total1 - 3) if all(target[j] in optimal1_set for j in range(i, i+4)))
        print(f"  File-specific mode-1 table: {optimal1_runs} 4-byte runs vs default {default1_runs} ({(optimal1_runs-default1_runs)/(total1-3)*100:.1f}% more coverage)")
    else:
        from expand import set_file_specific_mode1_table
        set_file_specific_mode1_table(None)

    # Build frequency-ranked byte table for FREQ_TABLE strategy
    freq_table = _build_freq_table(target)
    from expand import set_freq_table
    set_freq_table(freq_table)
    # Show top-15 coverage
    ft_freq = Counter(target)
    ft_top = sum(ft_freq.get(freq_table[i], 0) for i in range(len(freq_table)))
    print(f"  FREQ_TABLE v4 top-{len(freq_table)} coverage: {ft_top}/{tlen} ({ft_top/tlen*100:.1f}%)")

    # Build keyword table for KEYWORD_TABLE strategy
    keyword_table = _build_keyword_table(target)
    from expand import set_keyword_table
    set_keyword_table(keyword_table)
    if keyword_table:
        kw_total_bytes = sum(len(kw) for kw in keyword_table[:10])
        print(f"  KEYWORD_TABLE: {len(keyword_table)} keywords, top-10 total {kw_total_bytes}B")
    else:
        print(f"  KEYWORD_TABLE: no keywords found")

    # Build file-specific BPE pair table if it helps
    bpe_table = _build_optimal_bpe_table(target, min_improvement=0.05)
    if bpe_table:
        from expand import set_file_specific_bpe_table
        set_file_specific_bpe_table(bpe_table)
    else:
        from expand import set_file_specific_bpe_table
        set_file_specific_bpe_table(None)

    # Get V2 baseline -- but verify it actually covers the full file
    # Skip V2 baseline for large files (>100KB) -- V2 is slow and V3 always wins
    v2_seeds = None
    v2_count = 999
    v2_valid = False
    if tlen <= 100000:
        v2_seeds = _find_multi_seeds_dp(target, timeout * 0.15, max_seeds=128)
        v2_count = len(v2_seeds) if v2_seeds else 999
        if v2_seeds:
            v2_decoded = expand_multi(v2_seeds)
            v2_valid = (len(v2_decoded) >= tlen and v2_decoded[:tlen] == target)
    print(f"  V2 baseline: {v2_count} seeds (covers file: {v2_valid})")

    # Strategy: try encoding with and without setup seeds, pick best.
    # Also compare to V2 baseline -- never use more pixels than V2.
    # V2 is only a valid baseline if it actually covers the full file.
    # If V2 only covers a fraction (hit max_seeds cap), any full V3 encoding wins.
    INF = float('inf')
    best_total = v2_count if v2_valid else INF
    best_seeds_list = v2_seeds  # Store as a flat list for V2 fallback
    best_is_v3 = False
    best_setup_seeds = []
    best_data_seeds = None
    best_png = None

    # Helper: build a V2-compat PNG from seed list
    def _make_v2_png_fallback(seeds):
        from boot2 import make_multipixel_png
        return make_multipixel_png(seeds)

    # --- Option A: No setup seeds (pure LZ77 from natural output) ---
    # Allocate more time for large files (match enumeration scales ~linearly with size)
    if tlen > 500000:
        context_timeout = timeout * 0.70  # 70% for files > 500KB
    elif tlen > 100000:
        context_timeout = timeout * 0.50  # 50% for files > 100KB
    else:
        context_timeout = timeout * 0.30
    data_seeds_a = _encode_with_context(target, bytearray(), {},
                                         context_timeout, start_time)
    if data_seeds_a:
        total_a = len(data_seeds_a)
        if total_a <= best_total:
            png_a = make_v3_png(data_seeds_a, xor_mode=use_xor, dict_only=0,
                                bp8table=bp8table, bp_mode6_table=bp_mode6_table,
                                bp_mode1_table=bp_mode1_table,
                                freq_table=freq_table, keyword_table=keyword_table,
                                bpe_table=bpe_table)
            decoded_a = expand_from_png_v3(png_a)
            if decoded_a == target:
                best_total = total_a
                best_setup_seeds = []
                best_data_seeds = data_seeds_a
                best_png = png_a
                best_is_v3 = True
                print(f"  No-setup V3: {total_a} pixels (chosen)")

    # --- Option B: With setup seeds ---
    setup_patterns = _find_setup_candidates(target, max_setup_seeds=50)
    all_setup_seeds = []
    setup_buffer = bytearray()
    setup_ranges = {}

    for pattern, v1_seeds, count, savings in setup_patterns:
        all_setup_seeds.extend(v1_seeds)
        setup_buffer.extend(pattern)
        print(f"  Setup: {pattern!r} ({len(pattern)}B x{count}, saves ~{savings}px, {len(v1_seeds)} setup seeds)")
        pos = 0
        while True:
            idx = target.find(pattern, pos)
            if idx == -1:
                break
            setup_ranges[(idx, idx + len(pattern))] = True
            pos = idx + 1

    data_seeds_b = _encode_with_context(target, setup_buffer, setup_ranges,
                                         context_timeout, start_time)
    if data_seeds_b:
        total_b = len(all_setup_seeds) + len(data_seeds_b)
        if total_b <= best_total:
            png_b = make_v3_png(all_setup_seeds + data_seeds_b,
                                xor_mode=use_xor, dict_only=len(all_setup_seeds),
                                bp8table=bp8table, bp_mode6_table=bp_mode6_table,
                                bp_mode1_table=bp_mode1_table,
                                freq_table=freq_table, keyword_table=keyword_table,
                                bpe_table=bpe_table)
            decoded_b = expand_from_png_v3(png_b)
            if decoded_b == target:
                print(f"  With-setup V3: {total_b} pixels ({len(all_setup_seeds)} setup + {len(data_seeds_b)} data)")
                if total_b < best_total:
                    best_total = total_b
                    best_setup_seeds = all_setup_seeds
                    best_data_seeds = data_seeds_b
                    best_png = png_b
                    best_is_v3 = True

    # If no V3 option beat V2, use V2 (but only if V2 actually covers the file)
    if best_png is None or not best_is_v3:
        if v2_valid:
            print(f"  V2 fallback: {v2_count} pixels (V3 could not improve)")
            # Build a simple V3 PNG from V2 seeds (no context needed)
            from boot2 import make_multipixel_png
            best_png = make_multipixel_png(v2_seeds)
            best_setup_seeds = []
            best_data_seeds = v2_seeds
            best_total = v2_count
        else:
            print(f"  ERROR: No valid encoding found (V2 partial, V3 failed)")
            best_data_seeds = None
            best_png = None

    # Report results
    if best_data_seeds is None or best_png is None:
        print(f"  FAILED: Could not encode {tlen} bytes")
        from expand import (set_file_specific_table, set_file_specific_mode6_table,
                            set_file_specific_mode1_table, set_freq_table, set_keyword_table,
                            set_file_specific_bpe_table)
        set_file_specific_table(None)
        set_file_specific_mode6_table(None)
        set_file_specific_mode1_table(None)
        set_freq_table(None)
        set_keyword_table(None)
        set_file_specific_bpe_table(None)
        if output_png:
            return None, None
        return None, None

    dict_only = len(best_setup_seeds)
    all_seeds = best_setup_seeds + best_data_seeds
    width, height = _auto_dimensions(len(all_seeds))
    print(f"  V3 result: {tlen}B -> {len(all_seeds)} pixels ({width}x{height}) [{dict_only} setup + {len(best_data_seeds)} data]")
    saved = v2_count - len(all_seeds) if v2_valid else 0
    pct = (saved / v2_count * 100) if v2_count and v2_valid else 0
    print(f"  Saved: {saved} pixels ({pct:.0f}% reduction vs V2)")

    # Re-set tables for accurate strategy breakdown (they were cleared by expand_from_png_v3 verify)
    if bp8table:
        from expand import set_file_specific_table
        set_file_specific_table(bp8table)
    if bp_mode6_table:
        from expand import set_file_specific_mode6_table
        set_file_specific_mode6_table(bp_mode6_table)
    if bp_mode1_table:
        from expand import set_file_specific_mode1_table
        set_file_specific_mode1_table(bp_mode1_table)
    if freq_table:
        from expand import set_freq_table
        set_freq_table(freq_table)
    if keyword_table:
        from expand import set_keyword_table
        set_keyword_table(keyword_table)
    
    _show_strategy_breakdown(all_seeds, dict_only)

    # Reset file-specific tables after encoding
    from expand import (set_file_specific_table, set_file_specific_mode6_table,
                        set_file_specific_mode1_table, set_freq_table, set_keyword_table,
                        set_file_specific_bpe_table)
    set_file_specific_table(None)
    set_file_specific_mode6_table(None)
    set_file_specific_mode1_table(None)
    set_freq_table(None)
    set_keyword_table(None)
    set_file_specific_bpe_table(None)

    if output_png:
        with open(output_png, 'wb') as f:
            f.write(best_png)

    return best_data_seeds, best_png


def _encode_with_context(target, setup_buffer, setup_ranges, timeout, global_start):
    """Encode target using DP-based optimal parser (minimum seeds).

    Three phases:
      1. Enumerate all strategy matches at every position (V1 + LZ77)
         Buffer at position P = setup_buffer + target[0:P] (deterministic).
      2. DP shortest-path from 0 to tlen (each edge = 1 seed)
      3. Replay the path, verify, and return seed list

    Key insight: the output buffer at position P is always
    setup_buffer + target[0:P], regardless of which seeds produced it,
    because seeds emit left-to-right and must cover the target exactly.
    So LZ77 offsets computed against this buffer are always valid.
    
    Falls back to greedy left-to-right parsing if DP times out.
    
    Args:
        target: Bytes to encode
        setup_buffer: Pre-emitted bytes from setup seeds (for LZ77 references)
        setup_ranges: Dict of (start,end)->True for positions covered by setup
        timeout: Maximum time in seconds for THIS encoding step
        global_start: Encoding start time for timeout calculation
    
    Returns:
        List of seeds on success, None on failure
    """
    tlen = len(target)
    if tlen == 0:
        return []

    # Reset timeout clock for large files -- the table building phase
    # consumed time that shouldn't count against the DP budget
    local_start = time.time()
    elapsed = local_start - global_start
    if elapsed > timeout * 0.5:
        # Table building ate >50% of budget; reset with full timeout
        global_start = local_start - (timeout * 0.05)  # Pretend 5% elapsed

    elapsed = time.time() - global_start
    if elapsed > timeout * 0.9:
        return _encode_greedy(target, setup_buffer, timeout, global_start)

    # Build a rolling hash table for fast LZ77 matching
    # The "virtual buffer" at position P is: setup_buffer + target[0:P]
    full_buf = bytes(setup_buffer) + bytes(target)
    buf_offset = len(setup_buffer)  # target[0] is at this index in full_buf

    # Phase 1: Enumerate matches at every position
    match_time = timeout * 0.7
    matches = _enumerate_matches_fast(target, setup_buffer, full_buf, buf_offset,
                                       match_time, global_start)

    # Phase 2: DP shortest path
    seeds = _dp_shortest_path(target, matches, timeout, global_start)

    if seeds is not None:
        return seeds

    return _encode_greedy(target, setup_buffer, timeout, global_start)


def _enumerate_matches_fast(target, setup_buffer, full_buf, buf_offset,
                            timeout, global_start):
    """Enumerate all strategy matches at every position in target.

    Uses trigram hash-chain LZ77 for fast back-reference matching with
    bounded chain walks (max 32 steps per lookup). Returns
    matches[pos] = list of (length, seed, strategy_name).
    """
    tlen = len(target)
    matches = [[] for _ in range(tlen)]

    # --- Build trigram hash-chain for LZ77 ---
    # Hash: (data[i] << 10 ^ data[i+1] << 5 ^ data[i+2]) & 0x3FFF
    # Chain: head[hash] -> position -> prev[position] -> earlier position
    # Window: 32768 bytes (fits 16-bit offset field)
    MAX_CHAIN_WALK = 512
    HASH_SIZE = 16384  # 0x3FFF + 1
    WINDOW = 32768

    full_len = len(full_buf)
    head = [-1] * HASH_SIZE       # head[hash] = most recent position
    chain_prev = [-1] * full_len  # chain_prev[pos] = earlier pos with same hash

    # Build chain AND save per-position head snapshot so queries start
    # from the most recent valid entry (avoiding future-heavy chains)
    pos_chain_head = [-1] * tlen  # head[h] snapshot for each target position
    for i in range(full_len - 2):
        h = ((full_buf[i] << 10) ^ (full_buf[i + 1] << 5) ^ full_buf[i + 2]) & 0x3FFF
        chain_prev[i] = head[h]
        head[h] = i
        # If this position is in the target region, save the previous head
        # as the starting point for queries at this target position.
        # The chain head BEFORE inserting position i is the most recent
        # earlier position with this hash -- exactly what LZ77 needs.
        ti = i - buf_offset
        if 0 <= ti < tlen:
            pos_chain_head[ti] = chain_prev[i]  # head BEFORE this insertion

    # --- Build bigram hash chain for 2-byte LZ77 fallback ---
    BIGRAM_HASH_SIZE = 65536  # full 16-bit bigram keys
    bigram_head = [-1] * BIGRAM_HASH_SIZE
    bigram_chain = [-1] * full_len
    bigram_pos_head = [-1] * tlen  # per-position bigram head snapshot
    for i in range(full_len - 1):
        bh = ((full_buf[i] << 8) | full_buf[i + 1]) & 0xFFFF
        bigram_chain[i] = bigram_head[bh]
        bigram_head[bh] = i
        ti = i - buf_offset
        if 0 <= ti < tlen:
            bigram_pos_head[ti] = bigram_chain[i]

    # Pre-import verify to avoid repeated import overhead
    from find_seed import _pack_dict_seed, _verify as _seed_verify

    for pos in range(tlen):
        if time.time() - global_start > timeout:
            break

        remaining = tlen - pos
        suffix = target[pos:]

        # --- V1 strategies (stateless, fast) ---
        for n in range(1, 8):
            decomp = _try_prefix_decompose(suffix, n, DICTIONARY)
            if decomp:
                dlen = sum(len(DICTIONARY[i]) for i in decomp)
                seed = _pack_dict_seed(n, decomp)
                if _seed_verify(seed, target[pos:pos+dlen]):
                    matches[pos].append((dlen, seed, f"DICT_{n}"))

        # DICTX5
        decomp = _try_prefix_decompose(suffix, 5, DICTIONARY_EXT)
        if decomp and all(i < 32 for i in decomp):
            dlen = sum(len(DICTIONARY_EXT[i]) for i in decomp)
            params = sum((idx & 0x1F) << (5 * i) for i, idx in enumerate(decomp))
            seed = 0x80000000 | params
            if _seed_verify(seed, target[pos:pos+dlen]):
                matches[pos].append((dlen, seed, "DICTX5"))

        # BPE (0x9): byte-pair encoding
        pair_to_idx_local = _bpe_pair_to_idx()
        for n_pairs in range(min(4, remaining // 2), 0, -1):
            pair_len = n_pairs * 2
            if pair_len > remaining:
                continue
            indices = []
            valid = True
            for pi in range(n_pairs):
                pair = target[pos + pi*2 : pos + pi*2 + 2]
                idx = pair_to_idx_local.get(pair)
                if idx is None:
                    valid = False
                    break
                indices.append(idx)
            if not valid:
                continue
            params = 0
            for i in range(4):
                if i < n_pairs:
                    params |= (indices[i] & 0x7F) << (7 * i)
            seed = 0x90000000 | params
            if _seed_verify(seed, target[pos:pos+pair_len]):
                matches[pos].append((pair_len, seed, "BPE"))

        # DICTX7
        decomp = _try_prefix_decompose(suffix, 7, SUB_DICT)
        if decomp:
            dlen = sum(len(SUB_DICT[i]) for i in decomp)
            params = sum((idx & 0xF) << (4 * i) for i, idx in enumerate(decomp))
            seed = 0xA0000000 | params
            if _seed_verify(seed, target[pos:pos+dlen]):
                matches[pos].append((dlen, seed, "DICTX7"))

        # NIBBLE
        if remaining >= 7:
            nib = _try_nibble(suffix[:7])
            if nib:
                matches[pos].append((7, nib, "NIBBLE"))

        # BYTEPACK -- lengths 1-18 for DP (mode 0/1 support up to 18 bytes)
        # Only try > 5 when first byte repeats (fast pre-check)
        max_bp = min(18, remaining)
        for seg_len in range(max_bp, 0, -1):
            if seg_len > 5:
                # Quick pre-check: mode 0 repeat requires 4th+ bytes = first byte
                # mode 1 repeat requires specific XOR pattern
                # Only try if first byte repeats at least once after position 2
                if seg_len > 3 and target[pos + 3] != target[pos]:
                    continue
            seg = target[pos:pos + seg_len]
            seed = _quick_bytepack(seg)
            if seed:
                matches[pos].append((seg_len, seed, "BYTEPACK"))

        # --- FREQ_TABLE (strategy 0xB): variable-width byte encoding ---
        # Try longest segment first (max 7 bytes). Encoder handles bit budget.
        from expand import get_freq_table
        ft = get_freq_table()
        if ft is not None:
            for seg_len in range(min(7, remaining), 0, -1):
                seg = target[pos:pos + seg_len]
                ft_result = _try_freq_table_encode(seg, ft)
                if ft_result:
                    seed, encoded_len = ft_result
                    # Only add if we encoded the full requested length
                    if encoded_len == seg_len:
                        matches[pos].append((encoded_len, seed, "FREQ_TABLE"))
                        break  # Found longest match
                    # Partial match -- try shorter segments
                    if encoded_len >= 3:
                        matches[pos].append((encoded_len, seed, "FREQ_TABLE"))
                        break

        # --- KEYWORD_TABLE (strategy 0xD): keyword lookup encoding ---
        from expand import get_keyword_table
        kw_table = get_keyword_table()
        if kw_table is not None and remaining >= 3:
            # Try keyword matching at this position (up to remaining bytes)
            seg = target[pos:pos + min(40, remaining)]
            kw_result = _try_keyword_table_encode(seg, kw_table)
            if kw_result:
                seed, encoded_len = kw_result
                matches[pos].append((encoded_len, seed, "KEYWORD_TABLE"))
            # Also try hybrid: kw1 + literal + kw2
            hybrid_result = _try_keyword_hybrid_encode(seg, kw_table)
            if hybrid_result:
                seed, encoded_len = hybrid_result
                matches[pos].append((encoded_len, seed, "KEYWORD_TABLE"))

        # --- LZ77 matches via trigram hash-chain ---
        buf_len = buf_offset + pos
        best_lz77_len = 0
        best_lz77_offset = 0
        if buf_len >= 2 and remaining >= 3:

            # Use per-position chain head (avoids walking through future entries)
            cand = pos_chain_head[pos]
            steps = 0

            while cand >= 0 and steps < MAX_CHAIN_WALK:
                steps += 1

                # Safety: skip positions at or after current (shouldn't happen with snapshot)
                if cand >= buf_len:
                    cand = chain_prev[cand]
                    continue

                # Enforce window distance
                offset = buf_len - 1 - cand
                if offset >= WINDOW:
                    cand = chain_prev[cand]
                    continue

                # Verify trigram matches before extending
                if (full_buf[cand] == target[pos] and
                    full_buf[cand + 1] == target[pos + 1] and
                    full_buf[cand + 2] == target[pos + 2]):

                    # Extend match
                    match_len = 3  # trigram already matched
                    max_match = min(remaining, 0xFFF)
                    ci = cand + 3
                    ti = pos + 3
                    while match_len < max_match:
                        if ci < buf_len:
                            if full_buf[ci] == target[ti]:
                                match_len += 1
                                ci += 1
                                ti += 1
                            else:
                                break
                        else:
                            # Overlapping copy
                            wrap_pos = ci - buf_len
                            if wrap_pos < match_len and ti < tlen:
                                if target[pos + wrap_pos] == target[ti]:
                                    match_len += 1
                                    ci += 1
                                    ti += 1
                                else:
                                    break
                            else:
                                break

                    if match_len > best_lz77_len:
                        best_lz77_len = match_len
                        best_lz77_offset = offset

                cand = chain_prev[cand]

            if best_lz77_len >= 2:
                # Verify and add lengths for DP flexibility
                emitted = full_buf[:buf_len]
                if _verify_lz77(best_lz77_offset, best_lz77_len,
                                emitted, target[pos:pos + best_lz77_len]):
                    # Add key lengths for DP: longest, and intermediates
                    for length in range(2, best_lz77_len + 1):
                        seed = _make_lz77_seed(best_lz77_offset, length)
                        if seed:
                            matches[pos].append((length, seed, "LZ77"))

        # Also try 2-byte LZ77 for positions where trigram didn't find good matches
        # Use bigram hash chain for fast full-buffer lookup
        if buf_len >= 1 and remaining >= 2 and best_lz77_len < 2:
            bigram = (target[pos] << 8) | target[pos + 1]
            bigram_h = bigram & 0xFFFF  # 16-bit bigram hash
            cand = bigram_pos_head[pos]
            best_lz77_len = 0
            best_lz77_offset = 0
            steps = 0

            while cand >= 0 and steps < 128:
                steps += 1
                if cand >= buf_len:
                    cand = bigram_chain[cand] if cand < len(bigram_chain) else -1
                    continue

                offset = buf_len - 1 - cand
                if offset >= WINDOW:
                    cand = bigram_chain[cand] if cand < len(bigram_chain) else -1
                    continue

                # Verify 2-byte match and extend
                if full_buf[cand] == target[pos] and cand + 1 < len(full_buf) and full_buf[cand + 1] == target[pos + 1]:
                    match_len = 2
                    max_match = min(remaining, 0xFFF)
                    ci = cand + 2
                    ti = pos + 2
                    while match_len < max_match:
                        if ci < buf_len:
                            if full_buf[ci] == target[ti]:
                                match_len += 1
                                ci += 1
                                ti += 1
                            else:
                                break
                        else:
                            wrap_pos = ci - buf_len
                            if wrap_pos < match_len and ti < tlen:
                                if target[pos + wrap_pos] == target[ti]:
                                    match_len += 1
                                    ci += 1
                                    ti += 1
                                else:
                                    break
                            else:
                                break

                    if match_len > best_lz77_len:
                        best_lz77_len = match_len
                        best_lz77_offset = offset

                cand = bigram_chain[cand] if cand < len(bigram_chain) else -1

            if best_lz77_len >= 2:
                emitted = full_buf[:buf_len]
                if _verify_lz77(best_lz77_offset, best_lz77_len,
                                emitted, target[pos:pos + best_lz77_len]):
                    for length in range(2, best_lz77_len + 1):
                        seed = _make_lz77_seed(best_lz77_offset, length)
                        if seed:
                            matches[pos].append((length, seed, "LZ77"))

        # --- Explicit setup buffer search ---
        # Hash chains may not reach setup buffer entries (they're oldest).
        # Do a direct search of setup_buffer for LZ77 matches.
        setup_len = len(setup_buffer)
        if setup_len > 0 and buf_offset >= 2 and remaining >= 2 and best_lz77_len < remaining:
            for si in range(setup_len):
                if setup_buffer[si] != target[pos]:
                    continue
                # Extend match
                match_len = 0
                max_match = min(remaining, 0xFFF)
                ci = si
                ti = pos
                while match_len < max_match:
                    if ci < setup_len:
                        if setup_buffer[ci] == target[ti]:
                            match_len += 1
                            ci += 1
                            ti += 1
                        else:
                            break
                    else:
                        # Into target territory - use full_buf for extension
                        fbi = buf_offset + (ci - setup_len)
                        if fbi < buf_len and full_buf[fbi] == target[ti]:
                            match_len += 1
                            ci += 1
                            ti += 1
                        else:
                            break

                if match_len > best_lz77_len and match_len >= 2:
                    offset = buf_len - 1 - si
                    if offset < (1 << 16):
                        emitted = full_buf[:buf_len]
                        if _verify_lz77(offset, match_len,
                                        emitted, target[pos:pos + match_len]):
                            best_lz77_len = match_len
                            best_lz77_offset = offset

            if best_lz77_len >= 2:
                for length in range(2, best_lz77_len + 1):
                    seed = _make_lz77_seed(best_lz77_offset, length)
                    if seed:
                        matches[pos].append((length, seed, "LZ77_SETUP"))

        # Ensure every position has at least a 1-byte BYTEPACK match
        if not matches[pos]:
            max_bp_fb = min(18, remaining)
            for seg_len in range(max_bp_fb, 0, -1):
                if seg_len > 5 and seg_len > 3 and target[pos + 3] != target[pos]:
                    continue
                seg = target[pos:pos + seg_len]
                seed = _quick_bytepack(seg)
                if seed:
                    matches[pos].append((seg_len, seed, "BYTEPACK"))
                    break

        # Deduplicate by length -- prefer LZ77 over other strategies at same length
        seen_lens = {}
        for length, seed, name in matches[pos]:
            if length not in seen_lens:
                seen_lens[length] = (length, seed, name)
            elif name == 'LZ77' and seen_lens[length][2] != 'LZ77':
                # LZ77 is preferred over BYTEPACK/BPE/etc at same length
                # (same seed cost but better for future LZ77 references)
                seen_lens[length] = (length, seed, name)
        matches[pos] = sorted(seen_lens.values(), key=lambda x: -x[0])

    return matches


def _add_search_matches(matches, target, pos, remaining, timeout, global_start):
    """Use search() to find any strategy match at target[pos:].
    
    Only called when no long V1 match exists. Tries longest first.
    Catches RLE, XOR_CHAIN, LINEAR, TEMPLATE, etc.
    """
    import io, sys
    # Suppress search() verbose output
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        for seg_len in range(min(5, remaining), 0, -1):
            if time.time() - global_start > timeout:
                break
            seg = target[pos:pos + seg_len]
            results = seed_search(seg, timeout=0.02)
            if results:
                seed, name = results[0]
                matches[pos].append((seg_len, seed, name))
                break  # Found a match, stop
    finally:
        sys.stdout = old_stdout


def _add_search_matches_extended_fast(matches, target, pos, remaining,
                                       best_so_far, timeout, global_start):
    """Search for strategy matches longer than best_so_far at target[pos:].

    Tries RLE, XOR_CHAIN, LINEAR, TEMPLATE, etc. that the V1 enumeration
    might miss. Only tries lengths > best_so_far for efficiency.
    Uses short per-call timeouts for speed.
    """
    import io, sys
    max_try = min(20, remaining)
    if max_try <= best_so_far:
        return
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        for seg_len in range(max_try, best_so_far, -1):
            if time.time() - global_start > timeout:
                break
            seg = target[pos:pos + seg_len]
            results = seed_search(seg, timeout=0.03)
            if results:
                seed, name = results[0]
                from find_seed import _verify
                if _verify(seed, seg):
                    matches[pos].append((seg_len, seed, name))
                    break  # Found longest, done
    finally:
        sys.stdout = old_stdout


def _dp_shortest_path(target, matches, timeout, global_start, _retry_count=0):
    """Find minimum-seed encoding using DP shortest path.
    
    Each edge from position pos -> pos+length has cost 1 (one seed).
    BFS forward pass computes minimum seeds to reach each position,
    then backtrace reconstructs the optimal seed sequence.
    
    After finding the path, verifies each LZ77 seed against the actual
    expansion context. If verification fails, retries with bad LZ77
    matches removed (up to 3 retries).
    
    Args:
        target: Bytes to encode
        matches: matches[pos] = list of (length, seed, strategy_name) at each position
        timeout: Maximum time in seconds
        global_start: Time of encoding start (for timeout calculation)
        _retry_count: Internal retry counter for LZ77 verification failures
    
    Returns:
        List of seeds on success, None if no valid path found
    """
    tlen = len(target)
    INF = float('inf')

    # dp[pos] = minimum seeds to cover target[0:pos]
    dp = [INF] * (tlen + 1)
    dp[0] = 0
    parent = [None] * (tlen + 1)  # (length, seed, name)

    # Forward DP: for each position, try all matches
    for pos in range(tlen):
        if dp[pos] == INF:
            continue
        if time.time() - global_start > timeout:
            break

        for length, seed, name in matches[pos]:
            end = pos + length
            if end <= tlen and dp[pos] + 1 < dp[end]:
                dp[end] = dp[pos] + 1
                parent[end] = (pos, length, seed, name)

    # If DP didn't reach the end, try filling gaps with search()
    if dp[tlen] == INF:
        return _dp_with_search_fallback(target, matches, dp, parent,
                                        timeout, global_start)

    # Replay: walk backwards from tlen to 0
    seeds = []
    pos = tlen
    while pos > 0:
        if parent[pos] is None:
            # Gap -- shouldn't happen if dp[tlen] is finite, but handle it
            return _dp_with_search_fallback(target, matches, dp, parent,
                                            timeout, global_start)
        prev_pos, length, seed, name = parent[pos]
        seeds.append((prev_pos, length, seed, name))
        pos = prev_pos

    seeds.reverse()

    # Verify the replay produces correct bytes using context-aware expansion
    result = []
    for seg_pos, length, seed, name in seeds:
        expected = target[seg_pos:seg_pos + length]
        if name == 'LZ77':
            # LZ77 seeds need context verification
            ctx = ExpandContext()
            ctx.output_buffer = bytearray(target[:seg_pos])
            expanded = expand_with_context(seed, ctx)
            if expanded != expected:
                # Context mismatch -- remove bad LZ77 matches and retry
                if _retry_count < 3:
                    return _dp_retry_without_bad_lz77(
                        target, matches, timeout, global_start, _retry_count)
                return None
        elif name in ('FREQ_TABLE', 'KEYWORD_TABLE'):
            # FREQ_TABLE and KEYWORD_TABLE use global table state
            ctx_verify = ExpandContext()
            expanded = expand_with_context(seed, ctx_verify)
            if expanded != expected:
                return None
        else:
            if not _verify_v1(seed, expected):
                return None
        result.append(seed)

    print(f"  DP optimal: {len(result)} seeds (vs greedy baseline)")

    # Build correct seed_map from DP path data (before position info is lost)
    _strat_nibble = {"LZ77": 0xC, "BYTEPACK": 0xE, "BPE": 0x9, "FREQ_TABLE": 0x8,
                     "KEYWORD_TABLE": 0xA, "NIBBLE": 0x7, "RLE": 0xD, "DYN_DICT": 0xB,
                     "DICTX5": 0x5, "DICTX7": 0x6, "DICT_1": 0x1, "DICT_4": 0x4, "DICT_5": 0x5}
    dp_seed_map = [(seg_pos, seg_pos + length, seed, _strat_nibble.get(name, (seed >> 28) & 0xF))
                    for seg_pos, length, seed, name in seeds]

    # Post-DP consolidation: merge adjacent short seeds into LZ77
    result = _consolidate_seeds(target, result, dp_seed_map)
    if result:
        print(f"  After consolidation: {len(result)} seeds")

    return result


def _consolidate_seeds(target, seeds, dp_seed_map=None):
    """Post-DP optimization: replace short seed runs with LZ77 back-references.
    
    Three strategies applied in order:
      1. Single-seed replacement: if a non-LZ77 seed's bytes appear earlier in
         the emitted buffer, replace with LZ77 (same pixel cost, better for future refs)
      2. Run merging: if a run of 2-6 short non-LZ77 seeds' combined bytes appear
         earlier, replace entire run with one LZ77 seed (saves N-1 pixels)
      3. Partial run merging: try sub-runs within longer runs for best savings
    
    Args:
        target: the original target bytes
        seeds: list of seed values from DP
        dp_seed_map: pre-built seed_map with correct positions from DP path.
            If None, will attempt to rebuild (may be inaccurate for context-dependent seeds).
    """
    if not seeds or len(seeds) < 2:
        return seeds

    tlen = len(target)

    # Build position map: seed_idx -> (start_pos, end_pos, seed, strategy)
    if dp_seed_map is not None:
        seed_map = list(dp_seed_map)
    else:
        ctx = ExpandContext()
        seed_map = []
        pos = 0
        for s in seeds:
            result = expand_with_context(s, ctx)
            strat = (s >> 28) & 0xF
            seed_map.append((pos, pos + len(result), s, strat))
            pos += len(result)

    result_seeds = list(seeds)
    savings = 0

    # --- Phase 1: Replace individual non-LZ77 seeds with LZ77 where possible ---
    # Extended: seeds covering 2-18 bytes (was 2-5) since LZ77 handles up to 4095
    for i in range(len(seed_map)):
        sp, ep, s, st = seed_map[i]
        if st == 0xC:  # Already LZ77
            continue
        seg_len = ep - sp
        if seg_len > 18 or seg_len < 2:
            continue
        if sp < seg_len:  # Not enough buffer before this position
            continue

        data = target[sp:ep]
        search_buf = target[:sp]

        # Try to find these bytes earlier in the buffer
        found_at = search_buf.rfind(data)
        if found_at >= 0:
            offset = sp - 1 - found_at
            if offset < (1 << 16):
                lz77_seed = _make_lz77_seed(offset, seg_len)
                if lz77_seed:
                    verify_ctx = ExpandContext()
                    verify_ctx.output_buffer = bytearray(search_buf)
                    expanded = expand_with_context(lz77_seed, verify_ctx)
                    if expanded == data:
                        result_seeds[i] = lz77_seed
                        seed_map[i] = (sp, ep, lz77_seed, 0xC)
                        # No seed savings, but this byte range is now "known"
                        # and may help future consolidation

    # --- Phase 2: Merge runs of short non-LZ77 seeds into single LZ77 ---
    i = 0
    while i < len(seed_map):
        # Start a run at position i -- include seeds covering <= 18 bytes
        # and not already LZ77. Cap total run to 256B for fast rfind.
        j = i
        total_bytes = 0
        while j < len(seed_map):
            sp, ep, s, st = seed_map[j]
            if st == 0xC:  # Already LZ77
                break
            seg_bytes = ep - sp
            if seg_bytes > 18:  # Too long for run merging
                break
            if total_bytes + seg_bytes > 256:  # Cap run size for speed
                break
            total_bytes += seg_bytes
            j += 1

        run_len = j - i
        if run_len >= 2 and total_bytes >= 3:
            start_pos = seed_map[i][0]
            end_pos = seed_map[j - 1][1]
            combined = target[start_pos:end_pos]
            combined_len = len(combined)

            # Try full run first
            if start_pos >= combined_len and combined_len <= 0xFFF:
                search_buf = target[:start_pos]
                found_at = search_buf.rfind(combined)
                if found_at >= 0:
                    offset = start_pos - 1 - found_at
                    if offset < (1 << 16):
                        lz77_seed = _make_lz77_seed(offset, combined_len)
                        if lz77_seed:
                            verify_ctx = ExpandContext()
                            verify_ctx.output_buffer = bytearray(search_buf)
                            expanded = expand_with_context(lz77_seed, verify_ctx)
                            if expanded == combined:
                                result_seeds[i] = lz77_seed
                                for k in range(i + 1, j):
                                    result_seeds[k] = None
                                savings += run_len - 1
                                i = j
                                continue

            # Try partial sub-runs (shorter combinations)
            if run_len >= 3:
                best_sub_savings = 0
                best_sub_i = -1
                best_sub_j = -1
                best_sub_seed = None

                for si in range(i, j):
                    for sj in range(si + 2, j + 1):
                        sub_start = seed_map[si][0]
                        sub_end = seed_map[sj - 1][1]
                        sub_combined = target[sub_start:sub_end]
                        sub_len = len(sub_combined)
                        sub_seeds = sj - si

                        if sub_start >= sub_len and sub_len <= 0xFFF:
                            search_buf = target[:sub_start]
                            found_at = search_buf.rfind(sub_combined)
                            if found_at >= 0:
                                offset = sub_start - 1 - found_at
                                if offset < (1 << 16):
                                    lz77_seed = _make_lz77_seed(offset, sub_len)
                                    if lz77_seed:
                                        verify_ctx = ExpandContext()
                                        verify_ctx.output_buffer = bytearray(search_buf)
                                        expanded = expand_with_context(lz77_seed, verify_ctx)
                                        if expanded == sub_combined:
                                            sub_savings = sub_seeds - 1
                                            if sub_savings > best_sub_savings:
                                                best_sub_savings = sub_savings
                                                best_sub_i = si
                                                best_sub_j = sj
                                                best_sub_seed = lz77_seed

                if best_sub_savings > 0 and best_sub_seed is not None:
                    result_seeds[best_sub_i] = best_sub_seed
                    for k in range(best_sub_i + 1, best_sub_j):
                        result_seeds[k] = None
                    savings += best_sub_savings

        i += 1

    if savings > 0:
        result_seeds = [s for s in result_seeds if s is not None]

    return result_seeds


def _verify_v1(seed, expected):
    """Verify a V1 (non-LZ77) seed produces the expected bytes."""
    from find_seed import _verify
    return _verify(seed, expected)


def _dp_retry_without_bad_lz77(target, matches, timeout, global_start, retry_count):
    """Retry DP after removing LZ77 matches that fail context verification."""
    tlen = len(target)

    # Filter out bad LZ77 matches
    for pos in range(tlen):
        good = []
        for length, seed, name in matches[pos]:
            if name == 'LZ77':
                ctx = ExpandContext()
                ctx.output_buffer = bytearray(target[:pos])
                expanded = expand_with_context(seed, ctx)
                if expanded == target[pos:pos + length]:
                    good.append((length, seed, name))
            else:
                good.append((length, seed, name))
        matches[pos] = good

    # Re-run DP with incremented retry count
    return _dp_shortest_path(target, matches, timeout, global_start,
                             retry_count + 1)


def _dp_with_search_fallback(target, matches, dp, parent, timeout, global_start):
    """DP with search() fallback for uncovered positions.

    First fill gaps in matches[] using search(), then re-run DP.
    """
    tlen = len(target)
    INF = float('inf')

    # Find positions with no matches and fill with search()
    for pos in range(tlen):
        if matches[pos]:
            continue
        if time.time() - global_start > timeout:
            break

        remaining = tlen - pos
        for seg_len in range(min(20, remaining), 0, -1):
            if time.time() - global_start > timeout:
                break
            seg = target[pos:pos + seg_len]
            import io, sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                results = seed_search(seg, timeout=0.3)
            finally:
                sys.stdout = old_stdout
            if results:
                matches[pos].append((seg_len, results[0][0], results[0][1]))
                break

    # Re-run DP
    dp2 = [INF] * (tlen + 1)
    dp2[0] = 0
    parent2 = [None] * (tlen + 1)

    for pos in range(tlen):
        if dp2[pos] == INF:
            continue
        if time.time() - global_start > timeout:
            break

        if not matches[pos]:
            continue

        for length, seed, name in matches[pos]:
            end = pos + length
            if end <= tlen and dp2[pos] + 1 < dp2[end]:
                dp2[end] = dp2[pos] + 1
                parent2[end] = (pos, length, seed, name)

    if dp2[tlen] == INF:
        return None

    # Replay
    seeds = []
    pos = tlen
    while pos > 0:
        if parent2[pos] is None:
            return None
        prev_pos, length, seed, name = parent2[pos]
        seeds.append(seed)
        pos = prev_pos

    seeds.reverse()
    print(f"  DP+search: {len(seeds)} seeds")
    return seeds


def _encode_greedy(target, setup_buffer, timeout, global_start):
    """Original greedy left-to-right parser as fallback."""
    tlen = len(target)
    result_seeds = []
    emitted = bytearray(setup_buffer)
    pos = 0

    while pos < tlen:
        if time.time() - global_start > timeout:
            print(f"  Timeout at position {pos}/{tlen}")
            break

        remaining = tlen - pos
        best_len = 0
        best_seed = None

        if len(emitted) > 0:
            lz77 = _find_lz77_at(target, pos, emitted)
            if lz77:
                lz77_len, lz77_offset = lz77
                seed = _make_lz77_seed(lz77_offset, lz77_len)
                if seed and lz77_len > best_len:
                    best_len = lz77_len
                    best_seed = seed

        v1_match = _find_v1_match(target, pos)
        if v1_match and v1_match[0] > best_len:
            best_len, best_seed, _ = v1_match

        if best_len == 0:
            import io, sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                for seg_len in range(min(20, remaining), 0, -1):
                    if time.time() - global_start > timeout:
                        break
                    seg = target[pos:pos + seg_len]
                    results = seed_search(seg, timeout=0.3)
                    if results:
                        best_len = seg_len
                        best_seed = results[0][0]
                        break
            finally:
                sys.stdout = old_stdout

        if best_seed is None or best_len == 0:
            print(f"  FAIL at offset {pos}: 0x{target[pos]:02X}")
            return None

        result_seeds.append(best_seed)
        emitted.extend(target[pos:pos + best_len])
        pos += best_len

    if pos != tlen:
        return None

    return result_seeds


# ============================================================
# Diagnostics
# ============================================================

def _diagnose_mismatch(expected, got, data_seeds, setup_seeds):
    ctx = ExpandContext()
    # Process setup seeds
    for seed in setup_seeds:
        emit_dict_seed(seed, ctx)
    # Process data seeds
    pos = 0
    for i, seed in enumerate(data_seeds):
        result = expand_with_context(seed, ctx)
        if pos < len(expected):
            exp = expected[pos:pos+len(result)]
            if result != exp:
                print(f"  Divergence at data seed {i}: 0x{seed:08X}")
                print(f"    Expected: {exp!r}")
                print(f"    Got:      {result!r}")
                return
        pos += len(result)


def _show_strategy_breakdown(seeds, dict_only=0):
    names = {
        0:'DICT_1',1:'DICT_2',2:'DICT_3',3:'DICT_4',4:'DICT_5',
        5:'DICT_6',6:'DICT_7',7:'NIBBLE',8:'DICTX5',9:'BPE',
        0xA:'DICTX7',0xB:'FREQ_TBL',0xC:'LZ77',0xD:'KWORD_TBL',
        0xE:'BYTEPACK',0xF:'TEMPLATE'
    }
    counts = {}
    bytes_by = {}
    ctx = ExpandContext()
    for i, s in enumerate(seeds):
        if i < dict_only:
            result = emit_dict_seed(s, ctx)
        else:
            result = expand_with_context(s, ctx)
        strat = (s >> 28) & 0xF
        name = names.get(strat, f'?{strat:X}')
        counts[name] = counts.get(name, 0) + 1
        bytes_by[name] = bytes_by.get(name, 0) + len(result)
    print(f"  Strategy breakdown:")
    for name in sorted(counts.keys()):
        print(f"    {name:12s}: {counts[name]:3d} seeds, {bytes_by[name]:4d} bytes")


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Pixelpack Phase 3 - Context-Aware Encoder v3")
        print()
        print("Usage:")
        print("  python3 boot3.py encode <input_file> <output.png> [--xor]")
        print("  python3 boot3.py decode <input.png> [output_file]")
        print("  python3 boot3.py demo")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == 'encode':
        if len(sys.argv) < 4:
            print("Usage: python3 boot3.py encode <input_file> <output.png>")
            sys.exit(1)
        use_xor = '--xor' in sys.argv
        target = open(sys.argv[2], 'rb').read()
        seeds, png_data = encode_v3(target, sys.argv[3], timeout=120.0, use_xor=use_xor)
        sys.exit(0 if seeds else 1)

    elif cmd == 'decode':
        if len(sys.argv) < 3:
            print("Usage: python3 boot3.py decode <input.png> [output_file]")
            sys.exit(1)
        with open(sys.argv[2], 'rb') as f:
            png_data = f.read()
        result = expand_from_png_v3(png_data)
        print(f"Output: {len(result)} bytes")
        try:
            print(f"Text: {result.decode('ascii')!r}")
        except UnicodeDecodeError:
            print(f"Hex: {result.hex()}")
        if len(sys.argv) > 3:
            with open(sys.argv[3], 'wb') as f:
                f.write(result)
            import os
            os.chmod(sys.argv[3], 0o755)
            print(f"Written to: {sys.argv[3]}")

    elif cmd == 'demo':
        print("=" * 60)
        print("PIXELPACK PHASE 3 DEMO")
        print("=" * 60)
        print()
        fib = b'def fibonacci(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        a, b = 0, 1\n        for i in range(2, n + 1):\n            a, b = b, a + b\n        return b\n\nfor i in range(10):\n    print(f"fib({i}) = {fibonacci(i)}")\n'
        print(f"Target: fibonacci ({len(fib)} bytes)")
        print(f"V2 baseline: 56 pixels")
        print()
        seeds, png_data = encode_v3(fib, '/tmp/fib_v3.png', timeout=120.0)
        if seeds:
            print(f"\nV3: {len(seeds)} pixels (vs 56 V2)")

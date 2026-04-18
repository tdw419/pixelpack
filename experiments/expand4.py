"""
Pixelpack Phase 4 - Boot Pixel Architecture

Introduces the compute/display split. Boot pixels configure decoder state
without producing output bytes. Display pixels expand using configured state.

Boot pixel bit layout (strategy 0xF when in boot mode):
  [TTTT] [OOOO] [OOOO] [PPPPPPPPPPPPPPPPPPPPPPPP]
  0xF     opcode sub    24-bit payload

Boot mode is indicated by PNG metadata flag t6mode=1.
Boot pixels execute sequentially until BOOT_END is encountered.
After BOOT_END, remaining pixels are display pixels.

Backward compatibility: PNGs without t6mode decode via V3 path.

Phase 6a opcodes:
  0x0F: BOOT_END      -- end boot section, begin display
  0x3F: SET_PROFILE   -- load preset encoding configuration

Phase 6b opcodes:
  0x5F: SET_BPE_TABLE -- PRNG seed generates custom 127-entry byte-pair table

Phase 6c opcodes:
  0x7F: SET_TRANSFORM -- post-expansion byte transform (XOR, ADD, REVERSE, ROTATE)
"""

from dataclasses import dataclass, field
from expand import expand as expand_v1, seed_from_rgba, seed_to_rgba
from expand2 import expand_multi, expand_from_png, extract_seeds_from_png
from expand3 import (
    ExpandContext, expand_with_context, expand_multi_v3, expand_from_png_v3,
    _read_text_chunk,
)


# ============================================================
# Boot Context
# ============================================================

# Preset profiles -- configurations that tune decoder behavior
PROFILES = {
    0: {},  # Profile 0: default (no modifications)
    1: {'xor_mode': True},  # Profile 1: enable XOR channel
    # Future profiles can define custom BPE tables, strategy remaps, etc.
}


@dataclass
class BootContext:
    """Configuration state built by boot pixels."""
    profile_id: int = 0
    xor_mode: bool = False
    custom_bpe_table: list = None  # placeholder for phase 6b
    strategy_remap: dict = field(default_factory=dict)  # placeholder
    transform_type: int = 0  # phase 6c: 0=XOR, 1=ADD, 2=REVERSE, 3=ROTATE
    transform_param: int = 0  # phase 6c: transform parameter (0-255)

    def apply_profile(self, profile_id: int):
        """Load a preset profile. Resets to defaults then applies profile."""
        self.profile_id = profile_id
        # Reset to defaults first
        self.xor_mode = False
        # Apply profile overrides
        profile = PROFILES.get(profile_id, {})
        if 'xor_mode' in profile:
            self.xor_mode = profile['xor_mode']


# ============================================================
# Boot Instruction Decoder
# ============================================================

BOOT_END = 0x0       # opcode 0
BOOT_SET_PROFILE = 0x3  # opcode 3
BOOT_SET_BPE_TABLE = 0x5  # opcode 5
BOOT_SET_TRANSFORM = 0x7  # opcode 7

# Transform types (stored in BootContext.transform_type)
TRANSFORM_NONE = 0
TRANSFORM_XOR_CONST = 0   # XOR each byte with a constant
TRANSFORM_ADD_CONST = 1   # ADD a constant to each byte (mod 256)
TRANSFORM_REVERSE = 2     # Reverse the entire output
TRANSFORM_ROTATE = 3      # Circular shift output by N bytes


# ============================================================
# PRNG-Based BPE Table Generator
# ============================================================

# The "vocabulary" -- byte values that appear in programs.
# Weighted toward common ASCII: space, newline, letters, digits, symbols.
# The PRNG selects pairs from this pool.
_BYTE_POOL = bytes(range(32, 127))  # printable ASCII (95 chars)


def generate_bpe_table(prng_seed: int) -> list:
    """
    Deterministically generate a 128-entry BPE pair table from a PRNG seed.
    
    Index 0 = empty (terminator, same as fixed table).
    Indices 1-127 = byte pairs generated from the seed.
    
    Uses an LCG (Linear Congruential Generator) to produce pairs.
    Each pair is two bytes selected from the printable ASCII pool.
    
    The seed space is 12 bits (0-4095), giving 4096 possible tables.
    """
    # LCG parameters (same as Numerical Recipes)
    a = 1664525
    c = 1013904223
    state = prng_seed & 0xFFF  # 12-bit seed
    
    table = [b'']  # index 0 = empty/terminator
    
    # Generate 127 unique byte pairs
    seen = set()
    for _ in range(127 * 3):  # try up to 3x to fill table
        if len(table) >= 128:
            break
        state = (a * state + c) & 0xFFFFFFFF
        b1 = _BYTE_POOL[state % len(_BYTE_POOL)]
        state = (a * state + c) & 0xFFFFFFFF
        b2 = _BYTE_POOL[state % len(_BYTE_POOL)]
        pair = bytes([b1, b2])
        if pair not in seen:
            seen.add(pair)
            table.append(pair)
    
    # Pad if needed (shouldn't happen with 95^2 possible pairs)
    while len(table) < 128:
        table.append(b'')
    
    return table


def _decode_boot_opcode(seed: int) -> tuple:
    """
    Decode a boot pixel's opcode and payload.
    Returns (opcode_byte, payload_24bit) or None if not a boot pixel.

    Boot pixel format:
      strategy = 0xF (top 4 bits)
      params[27:24] = opcode (4 bits)
      params[23:0]  = payload (24 bits)

    We encode opcode as (opcode_nibble << 4) | sub_nibble for compact IDs.
    Phase 6a only uses two opcodes:
      BOOT_END:      params[27:24] = 0x0, params[23:0] = reserved
      SET_PROFILE:   params[27:24] = 0x3, params[23:0] = [4:profile_id][20:config_bits]
    """
    strategy = (seed >> 28) & 0xF
    if strategy != 0xF:
        return None  # not a boot pixel candidate

    params = seed & 0x0FFFFFFF
    opcode = (params >> 24) & 0xF
    payload = params & 0x00FFFFFF
    return (opcode, payload)


def _execute_boot_pixel(seed: int, boot_ctx: BootContext) -> str:
    """
    Execute a boot pixel instruction.
    Returns: 'boot_end' if BOOT_END encountered, 'continue' otherwise.
    """
    decoded = _decode_boot_opcode(seed)
    if decoded is None:
        # Non-0xF pixel in boot section -- treat as BOOT_END (auto-transition)
        return 'boot_end'

    opcode, payload = decoded

    if opcode == 0x0:
        # BOOT_END -- payload is reserved (ignore)
        return 'boot_end'

    elif opcode == 0x3:
        # SET_PROFILE
        profile_id = (payload >> 20) & 0xF
        config_bits = payload & 0x0FFFFF
        boot_ctx.apply_profile(profile_id)
        # Config bits reserved for future use
        return 'continue'

    elif opcode == 0x5:
        # SET_BPE_TABLE -- payload is 12-bit PRNG seed
        prng_seed = payload & 0xFFF
        boot_ctx.custom_bpe_table = generate_bpe_table(prng_seed)
        return 'continue'

    elif opcode == 0x7:
        # SET_TRANSFORM -- payload is [4:transform_type][8:param][12:reserved]
        transform_type = (payload >> 20) & 0xF
        transform_param = (payload >> 12) & 0xFF
        if transform_type <= TRANSFORM_ROTATE:
            boot_ctx.transform_type = transform_type
            boot_ctx.transform_param = transform_param
        # Unknown transform types: ignore silently (forward compat)
        return 'continue'

    else:
        # Unknown opcode -- ignore silently (forward compat)
        return 'continue'


# ============================================================
# V4 Expansion
# ============================================================

def _expand_bpe_with_table(seed: int, bpe_table: list) -> bytes:
    """
    Expand a BPE seed using a custom byte-pair table.
    Same logic as expand._expand_bpe but with a different table.
    """
    params = seed & 0x0FFFFFFF
    result = bytearray()
    for i in range(4):
        idx = (params >> (7 * i)) & 0x7F
        if idx == 0:
            break  # terminator
        if idx < len(bpe_table):
            pair = bpe_table[idx]
            if pair:
                result.extend(pair)
    return bytes(result)


# ============================================================
# PREDICT Strategy (Phase 8) -- Incremental Trigram Predictor
# ============================================================
#
# Strategy slot 0xA (repurposing DICTX7).
#
# Encoding: The 28-bit payload is a bitstream of unary-coded ranks.
#   Rank 0 (most likely byte):  0       (1 bit)
#   Rank 1:                     10      (2 bits)
#   Rank 2:                     110     (3 bits)
#   Rank N:                     N ones + 0 (N+1 bits)
#
# The decoder builds an order-3 trigram model from the already-decoded
# output buffer. No external table needed -- same principle as LZ77.
# The model updates incrementally (O(1) per byte decoded).
#
# With 78% of Python source bytes being rank-0 (1 bit each),
# a 28-bit seed averages ~16.5 bytes/seed -- 4x better than BYTEPACK.
#
# Activation: PNG metadata flag t8mode=1 (or strategy 0xA in V4 mode).

class PredictModel:
    """Incremental order-3 trigram prediction model.
    
    Updates O(1) per byte. Predictions use sorted frequency counts.
    Built entirely from the decode output buffer -- zero metadata overhead.
    """
    __slots__ = ('counts', 'history', '_dirty', '_cache_ctx', '_cache_ranked')
    
    def __init__(self, order=3):
        self.counts = {}       # bytes -> {int: int}  (context -> {byte: count})
        self.history = bytearray()
        self._dirty = False
        self._cache_ctx = None
        self._cache_ranked = None
    
    def add_byte(self, byte: int):
        """Update model with a newly decoded byte. O(1)."""
        h = self.history
        if len(h) >= 3:
            ctx = bytes(h[-3:])
            bucket = self.counts.get(ctx)
            if bucket is None:
                bucket = {}
                self.counts[ctx] = bucket
            bucket[byte] = bucket.get(byte, 0) + 1
            self._dirty = True
        h.append(byte)
    
    def predict(self, n=16):
        """Get top-N ranked predictions for the next byte. O(K log K) where K=unique bytes in context."""
        h = self.history
        if len(h) < 3:
            return []
        ctx = bytes(h[-3:])
        
        # Cache check (same context = no new data since last prediction)
        if not self._dirty and ctx == self._cache_ctx and self._cache_ranked is not None:
            return self._cache_ranked[:n]
        
        bucket = self.counts.get(ctx)
        if bucket is None:
            return []
        
        # Sort by count descending (most common first)
        ranked = sorted(bucket.keys(), key=lambda b: (-bucket[b], b))
        self._cache_ctx = ctx
        self._cache_ranked = ranked
        self._dirty = False
        return ranked[:n]


class PredictContext:
    """Wrapper that holds the PredictModel alongside V3's ExpandContext."""
    def __init__(self):
        self.predict_model = PredictModel()
        self.output_buffer = bytearray()


def _expand_predict(seed: int, model: PredictContext) -> bytes:
    """
    Expand a PREDICT seed (strategy 0xA).
    
    The 28-bit payload is a bitstream of unary-coded rank indices.
    Each rank selects a byte from the trigram model's predictions.
    
    Unary code: N ones followed by a zero = rank N.
    If we run out of bits (hit bit 28), the seed ends.
    """
    params = seed & 0x0FFFFFFF  # 28-bit payload
    result = bytearray()
    bit_pos = 0  # current bit position in payload
    
    while bit_pos < 28:
        # Read unary code: count consecutive 1s, then expect a 0
        rank = 0
        while bit_pos < 28:
            bit = (params >> bit_pos) & 1
            bit_pos += 1
            if bit == 0:
                break  # end of unary code
            rank += 1
        
        if bit_pos > 28 and rank > 0:
            # Ran out of bits mid-code -- incomplete, stop
            break
        
        # Look up byte at this rank in the model's predictions
        ranked = model.predict_model.predict()
        if rank >= len(ranked):
            break  # rank exceeds available predictions
        
        byte = ranked[rank]
        result.append(byte)
        model.predict_model.add_byte(byte)
        model.output_buffer.extend([byte])  # track in V3 context too
    
    return bytes(result)


def apply_transform(data: bytes, transform_type: int, transform_param: int) -> bytes:
    """
    Apply a post-expansion transform to decoded output.
    Called AFTER all display seeds have been expanded.
    """
    if transform_type == TRANSFORM_NONE or transform_type == TRANSFORM_XOR_CONST:
        # XOR_CONST: XOR each byte with transform_param
        if transform_param == 0:
            return data  # XOR 0 is identity
        return bytes([b ^ transform_param for b in data])

    elif transform_type == TRANSFORM_ADD_CONST:
        # ADD_CONST: add transform_param to each byte (mod 256)
        if transform_param == 0:
            return data
        return bytes([(b + transform_param) & 0xFF for b in data])

    elif transform_type == TRANSFORM_REVERSE:
        # REVERSE: reverse the entire output
        return data[::-1]

    elif transform_type == TRANSFORM_ROTATE:
        # ROTATE: circular left-shift by transform_param bytes
        if len(data) == 0:
            return data
        n = transform_param % len(data)
        if n == 0:
            return data
        return data[n:] + data[:n]

    else:
        # Unknown transform -- return data unchanged
        return data


def expand_multi_v4(seeds: list, max_output: int = 65536) -> bytes:
    """
    Expand seeds with boot pixel + PREDICT support.

    Scans for boot pixels (t6mode behavior without PNG). The first seed
    with strategy 0xF is treated as a potential boot pixel. Boot section
    ends at BOOT_END or first non-boot pixel.

    Strategy 0xA = PREDICT (incremental trigram predictor).
    Other strategies delegate to V3 expansion.
    """
    boot_ctx = BootContext()
    expand_ctx = ExpandContext()
    predict_ctx = PredictContext()
    result = bytearray()

    in_boot = True  # start in boot mode

    for seed in seeds:
        if len(result) >= max_output:
            break

        if in_boot:
            decoded = _decode_boot_opcode(seed)
            if decoded is not None:
                action = _execute_boot_pixel(seed, boot_ctx)
                if action == 'boot_end':
                    in_boot = False
                    # Apply boot context to expand context
                    expand_ctx.xor_mode = boot_ctx.xor_mode
                continue  # boot pixels never produce output
            else:
                # First non-0xF pixel = auto-transition to display
                in_boot = False
                expand_ctx.xor_mode = boot_ctx.xor_mode

        # Display pixel -- expand with context
        strategy = (seed >> 28) & 0xF
        
        if strategy == 0xA:
            # PREDICT strategy -- incremental trigram predictor
            expanded = _expand_predict(seed, predict_ctx)
            # Also update V3's expand context output buffer for LZ77 compatibility
            expand_ctx.output_buffer.extend(expanded)
        elif strategy == 0x9 and boot_ctx.custom_bpe_table is not None:
            # BPE with custom table
            expanded = _expand_bpe_with_table(seed, boot_ctx.custom_bpe_table)
            expand_ctx.output_buffer.extend(expanded)
            predict_ctx.predict_model.history.extend(expanded)
            for b in expanded:
                predict_ctx.output_buffer.append(b)
        else:
            expanded = expand_with_context(seed, expand_ctx)
            # Update PREDICT model with V3 expansion output
            for b in expanded:
                predict_ctx.predict_model.add_byte(b)
        
        result.extend(expanded)

    # Apply post-expansion transform
    result = bytearray(apply_transform(
        bytes(result[:max_output]),
        boot_ctx.transform_type,
        boot_ctx.transform_param,
    ))
    return bytes(result)


def expand_from_png_v4(png_data: bytes) -> bytes:
    """
    Expand a PNG with V4 boot pixel support.

    Reads t6mode flag. If present and =1, uses V4 expansion.
    Otherwise falls back to V3 (which falls back to V2).
    
    When t6mode=1, also checks for bp8table and bp_mode6_table tEXt chunks
    and sets them before expanding seeds (resets after).
    """
    t6mode = _read_text_chunk(png_data, 't6mode')

    if t6mode != '1':
        # Not a phase 4 PNG -- use V3 expansion
        return expand_from_png_v3(png_data)

    # Check for file-specific tables
    bp8table_hex = _read_text_chunk(png_data, 'bp8table')
    bp_mode6_hex = _read_text_chunk(png_data, 'bp_mode6_table')
    
    if bp8table_hex:
        from expand import set_file_specific_table
        try:
            table_str = bytes.fromhex(bp8table_hex).decode('latin-1')
            set_file_specific_table(table_str)
        except (ValueError, UnicodeDecodeError):
            pass
    if bp_mode6_hex:
        from expand import set_file_specific_mode6_table
        try:
            table_str = bytes.fromhex(bp_mode6_hex).decode('latin-1')
            set_file_specific_mode6_table(table_str)
        except (ValueError, UnicodeDecodeError):
            pass

    try:
        seeds, real_count, _tables = extract_seeds_from_png(png_data)
        real_seeds = seeds[:real_count]

        boot_ctx = BootContext()
        expand_ctx = ExpandContext()
        predict_ctx = PredictContext()
        result = bytearray()
        in_boot = True

        for seed in real_seeds:
            if in_boot:
                decoded = _decode_boot_opcode(seed)
                if decoded is not None:
                    action = _execute_boot_pixel(seed, boot_ctx)
                    if action == 'boot_end':
                        in_boot = False
                        expand_ctx.xor_mode = boot_ctx.xor_mode
                    continue
                else:
                    in_boot = False
                    expand_ctx.xor_mode = boot_ctx.xor_mode

            strategy = (seed >> 28) & 0xF
            
            if strategy == 0xA:
                # PREDICT strategy
                expanded = _expand_predict(seed, predict_ctx)
                expand_ctx.output_buffer.extend(expanded)
            elif strategy == 0x9 and boot_ctx.custom_bpe_table is not None:
                expanded = _expand_bpe_with_table(seed, boot_ctx.custom_bpe_table)
                expand_ctx.output_buffer.extend(expanded)
                predict_ctx.predict_model.history.extend(expanded)
            else:
                expanded = expand_with_context(seed, expand_ctx)
                for b in expanded:
                    predict_ctx.predict_model.add_byte(b)
            
            result.extend(expanded)

        # Apply post-expansion transform
        result = apply_transform(bytes(result), boot_ctx.transform_type, boot_ctx.transform_param)
        return result
    finally:
        # Reset file-specific tables
        if bp8table_hex:
            from expand import set_file_specific_table
            set_file_specific_table(None)
        if bp_mode6_hex:
            from expand import set_file_specific_mode6_table
            set_file_specific_mode6_table(None)


# ============================================================
# Boot Pixel Seed Construction Helpers
# ============================================================

def make_predict_seed(ranks: list) -> int:
    """
    Construct a PREDICT seed from a list of rank indices.
    
    Each rank is encoded as unary: N ones followed by a zero.
    The ranks are packed into the 28-bit payload LSB-first.
    
    ranks: list of integers (0-15), each is the rank of the predicted byte.
    Returns: 32-bit seed with strategy 0xA.
    """
    payload = 0
    bit_pos = 0
    
    for rank in ranks:
        # Write 'rank' ones
        for _ in range(rank):
            if bit_pos >= 28:
                break
            payload |= (1 << bit_pos)
            bit_pos += 1
        # Write one zero
        if bit_pos >= 28:
            break
        # zero is already 0, just advance
        bit_pos += 1
    
    return 0xA0000000 | (payload & 0x0FFFFFFF)


def make_boot_end_seed() -> int:
    """Construct a BOOT_END seed: strategy=0xF, opcode=0, payload=0."""
    return 0xF0000000  # 1111 0000 0000 0000 0000 0000 0000 0000


def make_set_profile_seed(profile_id: int, config_bits: int = 0) -> int:
    """
    Construct a SET_PROFILE seed.

    profile_id: 0-15 (4 bits)
    config_bits: 20 bits reserved for future use
    """
    if not (0 <= profile_id <= 15):
        raise ValueError(f"profile_id must be 0-15, got {profile_id}")
    payload = (profile_id << 20) | (config_bits & 0x0FFFFF)
    # strategy=0xF, opcode=0x3, payload
    return 0xF0000000 | (0x3 << 24) | payload


def make_set_bpe_table_seed(prng_seed: int) -> int:
    """
    Construct a SET_BPE_TABLE seed.

    prng_seed: 0-4095 (12 bits) -- seed for deterministic BPE table generation
    """
    if not (0 <= prng_seed <= 4095):
        raise ValueError(f"prng_seed must be 0-4095, got {prng_seed}")
    payload = prng_seed & 0xFFF
    return 0xF0000000 | (0x5 << 24) | payload


def make_set_transform_seed(transform_type: int, transform_param: int = 0) -> int:
    """
    Construct a SET_TRANSFORM seed.

    transform_type: 0-15 (4 bits)
        0 = XOR_CONST  -- XOR each output byte with transform_param
        1 = ADD_CONST   -- ADD transform_param to each output byte (mod 256)
        2 = REVERSE     -- reverse entire output (param ignored)
        3 = ROTATE      -- circular left-shift by transform_param bytes
    transform_param: 0-255 (8 bits) -- parameter for the transform
    """
    if not (0 <= transform_type <= 15):
        raise ValueError(f"transform_type must be 0-15, got {transform_type}")
    if not (0 <= transform_param <= 255):
        raise ValueError(f"transform_param must be 0-255, got {transform_param}")
    payload = (transform_type << 20) | (transform_param << 12)
    return 0xF0000000 | (0x7 << 24) | payload


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Pixelpack Phase 4 - Boot Pixel Architecture")
        print()
        print("Usage:")
        print("  python3 expand4.py <seed_hex> [<seed_hex2> ...]")
        print("  python3 expand4.py --png <file.png>")
        sys.exit(1)

    if sys.argv[1] == '--png':
        with open(sys.argv[2], 'rb') as f:
            png_data = f.read()
        result = expand_from_png_v4(png_data)
        print(f"Output: {len(result)} bytes")
        try:
            print(f"ASCII: {result.decode('ascii')!r}")
        except UnicodeDecodeError:
            print(f"Hex: {result.hex()}")
    else:
        seeds = [int(s, 16) for s in sys.argv[1:]]
        result = expand_multi_v4(seeds)
        print(f"Output: {len(result)} bytes")
        try:
            print(f"ASCII: {result.decode('ascii')!r}")
        except UnicodeDecodeError:
            print(f"Hex: {result.hex()}")

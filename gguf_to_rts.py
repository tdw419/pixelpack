#!/usr/bin/env python3
"""Convert any binary file (GGUF, ELF, etc.) to .rts.png pixel image.

Modes:
  raw    -- 4 bytes per RGBA pixel, Hilbert layout (standard PixelRTS)
  delta  -- delta-encode bytes before pixel packing (better for structured data)

Usage:
  python3 gguf_to_rts.py <input> <output.rts.png> [--mode raw|delta]
  python3 gguf_to_rts.py <input.rts.png> <output> --decode
"""
import struct, json, math, os, sys, hashlib
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

# ── GGUF Parser ──────────────────────────────────────────────

GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian

# GGUF value types
GGUF_TYPE_UINT8    = 0
GGUF_TYPE_INT8     = 1
GGUF_TYPE_UINT16   = 2
GGUF_TYPE_INT16    = 3
GGUF_TYPE_UINT32   = 4
GGUF_TYPE_INT32    = 5
GGUF_TYPE_FLOAT32  = 6
GGUF_TYPE_BOOL     = 7
GGUF_TYPE_STRING   = 8
GGUF_TYPE_ARRAY    = 9
GGUF_TYPE_UINT64   = 10
GGUF_TYPE_INT64    = 11
GGUF_TYPE_FLOAT64  = 12

# GGUF tensor types
GGUF_TENSOR_TYPES = {
    0: ("F32", 4),
    1: ("F16", 2),
    2: ("Q4_0", 0.5625),   # block_size=32, type_size=18
    3: ("Q4_1", 0.625),    # block_size=32, type_size=20
    6: ("Q5_0", 0.6875),
    7: ("Q5_1", 0.75),
    8: ("Q8_0", 1.0625),
    9: ("Q8_1", 1.125),
    10: ("Q2_K", 0.3125),  # approximate
    11: ("Q3_K", 0.4375),
    12: ("Q4_K", 0.5625),
    13: ("Q5_K", 0.6875),
    14: ("Q6_K", 0.8125),
    15: ("Q8_K", 1.0625),
}


def read_gguf_string(data, offset):
    """Read a GGUF string (uint64 length + bytes). Returns (string, new_offset)."""
    length = struct.unpack_from("<Q", data, offset)[0]
    offset += 8
    s = data[offset:offset+length].decode("utf-8", errors="replace")
    offset += length
    return s, offset


def read_gguf_value_typed(data, offset, vtype):
    """Read a GGUF value when type is already known (for array elements)."""
    if vtype == GGUF_TYPE_UINT8:
        return data[offset], offset + 1
    elif vtype == GGUF_TYPE_INT8:
        return struct.unpack_from("<b", data, offset)[0], offset + 1
    elif vtype == GGUF_TYPE_UINT16:
        return struct.unpack_from("<H", data, offset)[0], offset + 2
    elif vtype == GGUF_TYPE_INT16:
        return struct.unpack_from("<h", data, offset)[0], offset + 2
    elif vtype == GGUF_TYPE_UINT32:
        return struct.unpack_from("<I", data, offset)[0], offset + 4
    elif vtype == GGUF_TYPE_INT32:
        return struct.unpack_from("<i", data, offset)[0], offset + 4
    elif vtype == GGUF_TYPE_FLOAT32:
        return struct.unpack_from("<f", data, offset)[0], offset + 4
    elif vtype == GGUF_TYPE_BOOL:
        return data[offset] != 0, offset + 1
    elif vtype == GGUF_TYPE_STRING:
        return read_gguf_string(data, offset)
    elif vtype == GGUF_TYPE_UINT64:
        return struct.unpack_from("<Q", data, offset)[0], offset + 8
    elif vtype == GGUF_TYPE_INT64:
        return struct.unpack_from("<q", data, offset)[0], offset + 8
    elif vtype == GGUF_TYPE_FLOAT64:
        return struct.unpack_from("<d", data, offset)[0], offset + 8
    else:
        raise ValueError(f"Unknown GGUF value type: {vtype}")


def read_gguf_value(data, offset):
    """Read a single GGUF value (type prefix + data). Returns (value, new_offset)."""
    vtype = struct.unpack_from("<I", data, offset)[0]
    offset += 4

    if vtype == GGUF_TYPE_ARRAY:
        elem_type = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        count = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        arr = []
        for _ in range(count):
            val, offset = read_gguf_value_typed(data, offset, elem_type)
            arr.append(val)
        return arr, offset
    else:
        return read_gguf_value_typed(data, offset, vtype)


def parse_gguf(filepath):
    """Parse GGUF file, return metadata dict and section offsets."""
    with open(filepath, "rb") as f:
        data = f.read()

    magic = struct.unpack_from("<I", data, 0)[0]
    if magic != GGUF_MAGIC:
        return None, None, data, 0  # Not GGUF, return raw bytes

    version = struct.unpack_from("<I", data, 4)[0]
    n_tensors = struct.unpack_from("<Q", data, 8)[0]
    n_kv = struct.unpack_from("<Q", data, 16)[0]

    offset = 24

    # Parse KV metadata
    metadata = {}
    for _ in range(n_kv):
        key, offset = read_gguf_string(data, offset)
        value, offset = read_gguf_value(data, offset)
        metadata[key] = value

    # Parse tensor infos
    tensors = []
    for i in range(n_tensors):
        name, offset = read_gguf_string(data, offset)
        n_dims = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        dims = []
        for _ in range(n_dims):
            d = struct.unpack_from("<Q", data, offset)[0]
            offset += 8
            dims.append(d)
        ttype = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        toffset = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        type_name = GGUF_TENSOR_TYPES.get(ttype, (f"UNKNOWN({ttype})", 0))[0]
        tensors.append({
            "name": name,
            "n_dims": n_dims,
            "dims": dims,
            "type": type_name,
            "type_id": ttype,
            "offset": toffset,
        })

    # Align tensor data to page boundary
    alignment = metadata.get("general.alignment", 32)
    data_start = offset
    aligned_start = ((data_start + alignment - 1) // alignment) * alignment

    return metadata, tensors, data, aligned_start


# ── Delta Encoding ────────────────────────────────────────────

def delta_encode(data):
    """Delta-encode bytes: output[0] = input[0], output[i] = input[i] - input[i-1] mod 256."""
    arr = np.frombuffer(data, dtype=np.uint8).copy()
    result = np.empty_like(arr)
    result[0] = arr[0]
    result[1:] = (arr[1:].astype(np.int16) - arr[:-1].astype(np.int16)) % 256
    return result.astype(np.uint8).tobytes()


def delta_decode(data):
    """Reverse delta encoding: output[0] = input[0], output[i] = (output[i-1] + input[i]) mod 256."""
    arr = np.frombuffer(data, dtype=np.uint8).copy()
    result = np.empty_like(arr)
    result[0] = arr[0]
    cumsum = np.cumsum(arr.astype(np.int16)) % 256
    result = cumsum.astype(np.uint8)
    return result.tobytes()


# ── Hilbert Curve ─────────────────────────────────────────────

def hilbert_xy_array(grid_side, n):
    """Compute Hilbert (x,y) for indices 0..n-1 using vectorized approach."""
    order = int(math.log2(grid_side))
    indices = np.arange(n, dtype=np.uint64)

    x = np.zeros(n, dtype=np.uint64)
    y = np.zeros(n, dtype=np.uint64)

    for s in range(order):
        shift = 2 * s
        rx = (indices >> shift) & 1
        ry = ((indices >> shift) >> 1) & 1

        mask = (ry == 0)
        rx1 = mask & (rx == 1)

        s_val = 1 << s
        x[rx1] = s_val - 1 - x[rx1]
        y[rx1] = s_val - 1 - y[rx1]

        tmp = x[mask].copy()
        x[mask] = y[mask]
        y[mask] = tmp

        x += rx * s_val
        y += ry * s_val

    return x.astype(np.uint32), y.astype(np.uint32)


# ── Encode: Binary → .rts.png ─────────────────────────────────

def encode(input_path, output_path, mode="raw"):
    """Convert any binary file to .rts.png."""
    # Try GGUF parse for metadata
    metadata, tensors, data, data_start = parse_gguf(input_path)
    is_gguf = metadata is not None

    if is_gguf:
        print(f"GGUF detected: version={metadata.get('general.architecture', '?')}")
        print(f"  Tensors: {len(tensors)}, KV pairs: {len(metadata)}")
        for t in tensors[:5]:
            dims_str = "x".join(str(d) for d in t["dims"])
            print(f"    {t['name']}: {dims_str} ({t['type']})")
        if len(tensors) > 5:
            print(f"    ... and {len(tensors)-5} more")
    else:
        print(f"Raw binary mode: {os.path.getsize(input_path):,} bytes")

    raw_data = data
    sha256 = hashlib.sha256(raw_data).hexdigest()[:16]

    # Delta encode if requested
    if mode == "delta":
        print(f"Delta-encoding {len(raw_data):,} bytes...")
        pixel_data = delta_encode(raw_data)
    else:
        pixel_data = raw_data

    # Pad to 4-byte boundary
    remainder = len(pixel_data) % 4
    if remainder:
        pixel_data = pixel_data + b'\x00' * (4 - remainder)

    num_pixels = len(pixel_data) // 4
    
    # Use rectangular grid to minimize padding waste
    # For small files: square power-of-2 grid (GPU-friendly)
    # For large files: rectangular, near-square, minimal padding
    if num_pixels <= 4096 * 4096:
        # Small file: power-of-2 square grid
        grid_side = 1
        while grid_side * grid_side < num_pixels:
            grid_side *= 2
        grid_w = grid_h = grid_side
    else:
        # Large file: rectangular grid, width is power-of-2 for PNG row alignment
        # Find minimal grid close to square
        import math as _m
        target = num_pixels
        # Make width a nice round number (power of 2 or multiple of 1024)
        grid_w = 1
        while grid_w * grid_w < target:
            grid_w *= 2
        grid_h = (target + grid_w - 1) // grid_w
        # Ensure height is at least 1
        grid_h = max(grid_h, 1)

    total_pixels = grid_w * grid_h
    print(f"Grid: {grid_w}x{grid_h} ({total_pixels:,} pixels)")
    print(f"Utilization: {num_pixels/total_pixels*100:.1f}%")

    # Pad to fill grid
    pad_needed = total_pixels * 4 - len(pixel_data)
    if pad_needed > 0:
        pixel_data = pixel_data + b'\x00' * pad_needed

    # For large grids (>4096 wide), use linear raster scan (avoids OOM on Hilbert)
    use_hilbert = grid_w <= 4096

    if use_hilbert:
        linear_pixels = np.frombuffer(pixel_data[:total_pixels*4], dtype=np.uint8).reshape(-1, 4)
        print("Computing Hilbert curve...")
        hx, hy = hilbert_xy_array(grid_w, total_pixels)

        print("Building image...")
        img_array = np.zeros((grid_w, grid_w, 4), dtype=np.uint8)
        CHUNK = 2_000_000
        for start in range(0, total_pixels, CHUNK):
            end = min(start + CHUNK, total_pixels)
            img_array[hx[start:end], hy[start:end]] = linear_pixels[start:end]
    else:
        print("Linear layout (large model)...")
        img_array = np.frombuffer(pixel_data[:total_pixels*4], dtype=np.uint8).reshape(
            grid_h, grid_w, 4)

    # Build PNG metadata
    pnginfo = PngInfo()
    meta_dict = {
        "format": "rts_binary",
        "version": "2.0",
        "mode": mode,
        "grid_size": f"{grid_w}x{grid_h}",
        "grid_w": str(grid_w),
        "grid_h": str(grid_h),
        "layout": "hilbert" if use_hilbert else "linear",
        "data_size": str(len(raw_data)),
        "original_file": os.path.basename(input_path),
        "sha256": sha256,
    }
    if is_gguf:
        meta_dict["source_format"] = "gguf"
        meta_dict["n_tensors"] = str(len(tensors))
        if tensors:
            meta_dict["tensor_types"] = ",".join(set(t["type"] for t in tensors))

    for k, v in meta_dict.items():
        pnginfo.add_text(k, str(v))

    img = Image.fromarray(img_array, "RGBA")
    img.save(output_path, "PNG", pnginfo=pnginfo)

    fsize = os.path.getsize(output_path)
    print(f"\nOutput: {output_path}")
    print(f"  Size: {fsize:,} bytes ({fsize/1024/1024:.1f} MB)")
    print(f"  Ratio: {fsize/len(raw_data)*100:.1f}% of original")
    if mode == "delta":
        print(f"  Mode: delta-encoded (delta+dither → pixel)")

    # Also write sidecar JSON with tensor info
    if is_gguf:
        sidecar = output_path.replace(".rts.png", ".rts.meta.json")
        meta_out = {
            "format": "rts_binary",
            "version": "2.0",
            "mode": mode,
            "source_format": "gguf",
            "grid_size": f"{grid_w}x{grid_h}",
            "data_size": len(raw_data),
            "sha256_full": hashlib.sha256(raw_data).hexdigest(),
            "tensors": tensors,
            "gguf_metadata": {k: str(v) for k, v in metadata.items() if not isinstance(v, (list, dict))},
        }
        with open(sidecar, "w") as f:
            json.dump(meta_out, f, indent=2, default=str)
        print(f"  Sidecar: {sidecar}")

    return True


# ── Decode: .rts.png → Binary ─────────────────────────────────

def decode(input_path, output_path):
    """Convert .rts.png back to original binary file."""
    # Allow large images (models can be gigabytes)
    Image.MAX_IMAGE_PIXELS = None
    
    img = Image.open(input_path)
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    # Read PNG metadata
    meta = {}
    for k, v in img.text.items():
        meta[k] = v

    mode = meta.get("mode", "raw")
    layout = meta.get("layout", "hilbert")
    data_size = int(meta.get("data_size", "0"))
    grid_w = int(meta.get("grid_w", str(img.width)))
    grid_h = int(meta.get("grid_h", str(img.height)))
    # Fallback for old format square grids
    if "grid_w" not in meta and "grid_size" in meta:
        gs = meta["grid_size"]
        try:
            w, h = gs.split("x")
            grid_w, grid_h = int(w), int(h)
        except:
            grid_w = grid_h = int(gs)

    print(f"Mode: {mode}, Layout: {layout}")
    print(f"Expected data size: {data_size:,} bytes")

    # Extract pixels
    img_array = np.array(img, dtype=np.uint8)
    total_pixels = grid_w * grid_h

    if layout == "linear":
        # Direct reshape -- bytes in raster order
        print("Extracting pixel data (linear)...")
        raw_pixels = img_array.reshape(-1).tobytes()
    else:
        # Hilbert inverse
        print("Computing inverse Hilbert curve...")
        hx, hy = hilbert_xy_array(grid_w, total_pixels)

        # Read pixels back in Hilbert order
        print("Extracting pixel data...")
        linear = np.zeros((total_pixels, 4), dtype=np.uint8)
        CHUNK = 2_000_000
        for start in range(0, total_pixels, CHUNK):
            end = min(start + CHUNK, total_pixels)
            linear[start:end] = img_array[hx[start:end], hy[start:end]]

        raw_pixels = linear.reshape(-1).tobytes()

    # Truncate to original data size
    if data_size > 0:
        pixel_data = raw_pixels[:data_size]
    else:
        # Strip trailing zeros (best effort)
        pixel_data = raw_pixels.rstrip(b'\x00')

    # Delta decode if needed
    if mode == "delta":
        print("Delta-decoding...")
        result = delta_decode(pixel_data)
    else:
        result = pixel_data

    # Verify
    sha256_expected = meta.get("sha256", "")
    sha256_actual = hashlib.sha256(result).hexdigest()[:16]
    match = sha256_actual == sha256_expected
    print(f"SHA256: {sha256_actual} {'MATCH' if match else 'MISMATCH!'}")

    with open(output_path, "wb") as f:
        f.write(result)

    print(f"Output: {output_path} ({len(result):,} bytes)")
    return match


# ── Entropy Analysis ──────────────────────────────────────────

def analyze(input_path):
    """Analyze binary file entropy by section (useful for choosing encoding mode)."""
    metadata, tensors, data, data_start = parse_gguf(input_path)
    is_gguf = metadata is not None

    print(f"File: {input_path} ({len(data):,} bytes)")
    print()

    if is_gguf:
        # Analyze header, metadata, tensor sections
        header_end = 24
        meta_end = data_start

        sections = [
            ("Header", data[:header_end]),
            ("Metadata", data[header_end:meta_end]),
        ]

        # Per-tensor sections
        for t in tensors:
            # Estimate tensor size from dims and type
            n_elements = 1
            for d in t["dims"]:
                n_elements *= d
            type_info = GGUF_TENSOR_TYPES.get(t["type_id"], ("?", 1))
            bytes_per = type_info[1] if isinstance(type_info[1], (int, float)) else 1
            tsize = int(n_elements * bytes_per)
            toff = data_start + t["offset"]
            sections.append((f"Tensor: {t['name']}", data[toff:toff+tsize]))

        print(f"{'Section':<40} {'Bytes':>12} {'Entropy':>8}")
        print("-" * 65)
        for name, section_data in sections:
            if len(section_data) == 0:
                continue
            # Shannon entropy
            counts = np.bincount(np.frombuffer(section_data, dtype=np.uint8), minlength=256)
            probs = counts[counts > 0] / len(section_data)
            entropy = -np.sum(probs * np.log2(probs))
            print(f"{name:<40} {len(section_data):>12,} {entropy:>8.3f}")

        # Delta analysis on first tensor
        for t in tensors:
            toff = data_start + t["offset"]
            n_elements = 1
            for d in t["dims"]:
                n_elements *= d
            type_info = GGUF_TENSOR_TYPES.get(t["type_id"], ("?", 1))
            bytes_per = type_info[1] if isinstance(type_info[1], (int, float)) else 1
            tsize = int(n_elements * bytes_per)
            if tsize < 1:
                continue

            raw_section = data[toff:toff+tsize]
            delta_section = delta_encode(raw_section)

            raw_counts = np.bincount(np.frombuffer(raw_section, dtype=np.uint8), minlength=256)
            raw_probs = raw_counts[raw_counts > 0] / len(raw_section)
            raw_entropy = -np.sum(raw_probs * np.log2(raw_probs))

            delta_counts = np.bincount(np.frombuffer(delta_section, dtype=np.uint8), minlength=256)
            delta_probs = delta_counts[delta_counts > 0] / len(delta_section)
            delta_entropy = -np.sum(delta_probs * np.log2(delta_probs))

            print(f"\n  Delta analysis for '{t['name']}':")
            print(f"    Raw entropy:   {raw_entropy:.3f} bits/byte")
            print(f"    Delta entropy: {delta_entropy:.3f} bits/byte")
            print(f"    Improvement:   {raw_entropy - delta_entropy:.3f} bits/byte ({(raw_entropy-delta_entropy)/raw_entropy*100:.1f}%)")
            break  # Just first tensor for now
    else:
        # Raw file analysis
        raw_data = data
        raw_counts = np.bincount(np.frombuffer(raw_data, dtype=np.uint8), minlength=256)
        raw_probs = raw_counts[raw_counts > 0] / len(raw_data)
        raw_entropy = -np.sum(raw_probs * np.log2(raw_probs))

        delta_data = delta_encode(raw_data)
        delta_counts = np.bincount(np.frombuffer(delta_data, dtype=np.uint8), minlength=256)
        delta_probs = delta_counts[delta_counts > 0] / len(delta_data)
        delta_entropy = -np.sum(delta_probs * np.log2(delta_probs))

        print(f"Raw entropy:   {raw_entropy:.3f} bits/byte")
        print(f"Delta entropy: {delta_entropy:.3f} bits/byte")
        print(f"Improvement:   {raw_entropy - delta_entropy:.3f} bits/byte ({(raw_entropy-delta_entropy)/raw_entropy*100:.1f}%)")


# ── CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print(f"  {sys.argv[0]} <input> <output.rts.png> [--mode raw|delta]")
        print(f"  {sys.argv[0]} <input.rts.png> <output> --decode")
        print(f"  {sys.argv[0]} <input> --analyze")
        sys.exit(1)

    if "--analyze" in sys.argv:
        analyze(sys.argv[1])
    elif "--decode" in sys.argv:
        decode(sys.argv[1], sys.argv[2])
    else:
        mode = "raw"
        if "--mode" in sys.argv:
            mode = sys.argv[sys.argv.index("--mode") + 1]
        encode(sys.argv[1], sys.argv[2], mode=mode)

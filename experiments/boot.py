"""
Boot Pixel - PNG Encoder/Decoder

Encodes a 32-bit seed as a 1x1 RGBA PNG and decodes it back.
No external dependencies - uses raw PNG chunk construction.
"""

import struct
import zlib
import sys
from expand import expand, seed_from_rgba, seed_to_rgba


def make_1x1_png(r: int, g: int, b: int, a: int) -> bytes:
    """Create a 1x1 RGBA PNG pixel."""
    def chunk(chunk_type, data):
        c = chunk_type + data
        crc = zlib.crc32(c) & 0xFFFFFFFF
        return struct.pack('>I', len(data)) + c + struct.pack('>I', crc)

    signature = b'\x89PNG\r\n\x1a\n'

    # IHDR: width=1, height=1, bit_depth=8, color_type=6 (RGBA), ...
    ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 6, 0, 0, 0)
    ihdr = chunk(b'IHDR', ihdr_data)

    # IDAT: filter byte (0=none) + RGBA pixel
    raw_data = bytes([0, r, g, b, a])
    compressed = zlib.compress(raw_data)
    idat = chunk(b'IDAT', compressed)

    iend = chunk(b'IEND', b'')

    return signature + ihdr + idat + iend


def read_png_pixel(png_data: bytes) -> tuple:
    """Read RGBA values from a 1x1 PNG. Returns (r, g, b, a)."""
    # Verify PNG signature
    if png_data[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError("Not a valid PNG file")

    pos = 8
    idat_data = b''
    width = height = 0

    while pos < len(png_data):
        length = struct.unpack('>I', png_data[pos:pos+4])[0]
        chunk_type = png_data[pos+4:pos+8]
        data = png_data[pos+8:pos+8+length]

        if chunk_type == b'IHDR':
            width, height = struct.unpack('>II', data[:8])
            if width != 1 or height != 1:
                raise ValueError(f"Expected 1x1 PNG, got {width}x{height}")

        elif chunk_type == b'IDAT':
            idat_data += data

        pos += 12 + length  # 4(len) + 4(type) + data + 4(crc)

    if not idat_data:
        raise ValueError("No IDAT chunk found in PNG")

    decompressed = zlib.decompress(idat_data)
    # Format: filter_byte + R + G + B + A
    if len(decompressed) < 5:
        raise ValueError(f"Decompressed data too short: {len(decompressed)}")

    _, r, g, b, a = decompressed[:5]
    return r, g, b, a


def encode_file(target_path: str, output_png: str, timeout: float = 60.0):
    """Encode a file into a 1x1 PNG boot pixel."""
    from find_seed import search

    with open(target_path, 'rb') as f:
        target = f.read()

    print(f"Encoding: {target_path} ({len(target)} bytes)")
    print(f"Target content: {target[:100]!r}{'...' if len(target) > 100 else ''}")
    print()

    results = search(target, timeout=timeout)
    if not results:
        print("FAILED: Could not find a seed for this target.")
        print("  Try a shorter target or update the dictionary.")
        return False

    seed, strategy_name = results[0]
    r, g, b, a = seed_to_rgba(seed)

    png_data = make_1x1_png(r, g, b, a)
    with open(output_png, 'wb') as f:
        f.write(png_data)

    print(f"\nEncoded into: {output_png}")
    print(f"  Pixel RGBA: ({r}, {g}, {b}, {a})")
    print(f"  Seed: 0x{seed:08X}")
    print(f"  Strategy: {strategy_name}")
    print(f"  PNG size: {len(png_data)} bytes")
    return True


def decode_png(png_path: str, output_path: str = None):
    """Decode a 1x1 PNG boot pixel back to bytes."""
    with open(png_path, 'rb') as f:
        png_data = f.read()

    r, g, b, a = read_png_pixel(png_data)
    seed = seed_from_rgba(r, g, b, a)

    print(f"Decoding: {png_path}")
    print(f"  Pixel RGBA: ({r}, {g}, {b}, {a})")
    print(f"  Seed: 0x{seed:08X}")

    result = expand(seed)
    print(f"  Output: {len(result)} bytes")
    try:
        print(f"  Text: {result.decode('ascii')!r}")
    except UnicodeDecodeError:
        print(f"  Hex: {result.hex()}")

    if output_path:
        with open(output_path, 'wb') as f:
            f.write(result)
        print(f"  Written to: {output_path}")
        # Try to make it executable
        import os
        os.chmod(output_path, 0o755)

    return result


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Boot Pixel - PNG Encoder/Decoder")
        print()
        print("Usage:")
        print("  python boot.py encode <input_file> <output.png>")
        print("  python boot.py decode <input.png> [output_file]")
        print("  python boot.py demo")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == 'encode':
        if len(sys.argv) < 4:
            print("Usage: python boot.py encode <input_file> <output.png>")
            sys.exit(1)
        success = encode_file(sys.argv[2], sys.argv[3])
        sys.exit(0 if success else 1)

    elif cmd == 'decode':
        if len(sys.argv) < 3:
            print("Usage: python boot.py decode <input.png> [output_file]")
            sys.exit(1)
        output = sys.argv[3] if len(sys.argv) > 3 else None
        result = decode_png(sys.argv[2], output)
        sys.exit(0)

    elif cmd == 'demo':
        print("=" * 60)
        print("BOOT PIXEL DEMO")
        print("=" * 60)
        print()

        # Create a target program
        target = b'print("Hello")\n'
        target_file = '/tmp/boot_pixel_target.py'
        with open(target_file, 'w') as f:
            f.write(target.decode())

        print(f"Target program: {target.decode()!r}")
        print(f"Saved to: {target_file}")
        print()

        # Encode
        png_path = '/tmp/boot_pixel.png'
        if not encode_file(target_file, png_path):
            print("Demo failed!")
            sys.exit(1)

        print()
        print("-" * 60)

        # Decode
        decoded_file = '/tmp/boot_pixel_decoded.py'
        decoded = decode_png(png_path, decoded_file)

        print()
        print("-" * 60)

        # Verify round-trip
        print("ROUND-TRIP VERIFICATION:")
        if decoded == target:
            print("  PASS: Decoded output matches original target!")
            print(f"  Original:  {target.hex()}")
            print(f"  Decoded:   {decoded.hex()}")
        else:
            print("  FAIL: Decoded output does NOT match!")
            print(f"  Original:  {target.hex()}")
            print(f"  Decoded:   {decoded.hex()}")
            sys.exit(1)

        print()
        print("Executing the decoded program:")
        print("-" * 60)
        import subprocess
        result = subprocess.run(
            [sys.executable, decoded_file],
            capture_output=True, text=True
        )
        print(result.stdout, end='')
        if result.stderr:
            print(f"STDERR: {result.stderr}", end='')
        print("-" * 60)
        print()
        print(f"Boot pixel PNG saved at: {png_path}")
        print(f"The pixel color encodes a complete Python program.")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

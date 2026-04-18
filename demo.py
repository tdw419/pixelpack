#!/usr/bin/env python3
"""Quick demo: create a small test file, encode as .rts.png, decode, verify SHA256.

Usage:
    python3 demo.py
"""
import os, sys, hashlib, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_to_rts import encode, decode


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    # Create a test file with recognizable content
    tmpdir = tempfile.mkdtemp(prefix="pixelpack_demo_")
    original = os.path.join(tmpdir, "test_data.bin")
    encoded = os.path.join(tmpdir, "test_data.rts.png")
    decoded = os.path.join(tmpdir, "test_data_decoded.bin")

    # Write ~1MB of mixed content
    with open(original, "wb") as f:
        # Some ASCII text
        f.write(b"# pixelpack demo test file\n" * 100)
        # Some structured data (float32 weights)
        import struct
        for i in range(10000):
            f.write(struct.pack("<f", i * 0.001))
        # Some random bytes
        import random
        random.seed(42)
        f.write(bytes(random.getrandbits(8) for _ in range(100000)))
        # Padding to ~1MB
        f.write(b"\x00" * (1048576 - f.tell()))

    orig_hash = sha256_file(original)
    orig_size = os.path.getsize(original)

    print(f"Test file: {orig_size:,} bytes")
    print(f"SHA256:    {orig_hash[:16]}...")

    # Encode
    print(f"\nEncoding → {encoded}")
    encode(original, encoded, mode="raw")
    png_size = os.path.getsize(encoded)
    print(f"PNG size:  {png_size:,} bytes ({png_size/orig_size*100:.1f}% of original)")

    # Decode
    print(f"\nDecoding → {decoded}")
    ok = decode(encoded, decoded)

    # Verify
    dec_hash = sha256_file(decoded)
    match = orig_hash == dec_hash

    print(f"\n{'PASS' if match else 'FAIL'}: SHA256 match = {match}")
    if match:
        print(f"Byte-perfect round-trip confirmed.")
        print(f"\nOpen {encoded} in any image viewer to see your data as pixels!")

    # Cleanup
    for f in [original, encoded, decoded]:
        os.unlink(f)
    os.rmdir(tmpdir)

    return 0 if match else 1


if __name__ == "__main__":
    sys.exit(main())

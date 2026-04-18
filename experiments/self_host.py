"""
Pixelpack Self-Hosting Encoder

Encodes all pixelpack source files into PNG images.
The resulting directory can be decoded by bootstrap.py to reconstruct
the entire pixelpack toolchain from nothing but PNG images.

Usage:
    python3 self_host.py              # encode all source -> self_host_pngs/
    python3 self_host.py <dir>        # encode all source -> <dir>/
    python3 self_host.py --verify     # encode + verify round-trip
"""

import os
import sys
import time
import json

# The source files that make up pixelpack's core toolchain
# Order matters: expand.py must be first (it's the foundation)
SOURCE_FILES = [
    'expand.py',     # The decoder - seed expansion engine
    'expand2.py',    # Multi-pixel expansion + PNG decode
    'find_seed.py',  # Seed search engine
    'boot2.py',      # Multi-pixel encoder/decoder + DP segmentation
    'self_host.py',  # Self-hosting encoder
    'bootstrap.py',  # Self-hosting decoder
]

# Optional files (verify, boot3/4, etc) - encode if present
OPTIONAL_FILES = [
    'boot.py',
    'boot3.py',
    'expand3.py',
    'expand4.py',
    'verify.py',
    'verify2.py',
    'verify3.py',
    'verify4.py',
]


def encode_source_files(output_dir, files_to_encode, verify=True):
    """Encode source files into PNG images in output_dir.
    
    Each file becomes <filename>.px.png in the output directory.
    A manifest.json records filenames, sizes, and checksums.
    """
    from boot2 import encode_multi, expand_from_png
    
    os.makedirs(output_dir, exist_ok=True)
    
    manifest = {
        'version': '1.0',
        'files': [],
        'total_source_bytes': 0,
        'total_png_bytes': 0,
    }
    
    t0 = time.time()
    
    for fname in files_to_encode:
        src_path = os.path.join(os.path.dirname(__file__), fname)
        if not os.path.exists(src_path):
            print(f"  SKIP {fname} (not found)")
            continue
        
        with open(src_path, 'rb') as f:
            source_data = f.read()
        
        png_name = fname + '.px.png'
        png_path = os.path.join(output_dir, png_name)
        
        print(f"\n{'='*60}")
        print(f"Encoding: {fname} ({len(source_data)} bytes)")
        print(f"{'='*60}")
        
        ok = encode_multi(source_data, png_path, timeout=0)
        
        if not ok:
            print(f"FAILED to encode {fname}")
            return False
        
        # Verify round-trip
        with open(png_path, 'rb') as f:
            png_data = f.read()
        
        decoded = expand_from_png(png_data)
        
        if decoded != source_data:
            print(f"ROUND-TRIP FAIL for {fname}")
            print(f"  Expected {len(source_data)} bytes, got {len(decoded)}")
            # Find first difference
            for i in range(min(len(decoded), len(source_data))):
                if decoded[i] != source_data[i]:
                    print(f"  First diff at byte {i}: expected 0x{source_data[i]:02X}, got 0x{decoded[i]:02X}")
                    break
            return False
        
        file_info = {
            'name': fname,
            'png': png_name,
            'source_bytes': len(source_data),
            'png_bytes': len(png_data),
        }
        manifest['files'].append(file_info)
        manifest['total_source_bytes'] += len(source_data)
        manifest['total_png_bytes'] += len(png_data)
        
        ratio = len(png_data) / len(source_data)
        print(f"\n  {fname}: {len(source_data)}B -> {len(png_data)}B PNG (ratio: {ratio:.2f}x)")
    
    elapsed = time.time() - t0
    
    # Write manifest
    manifest['elapsed_seconds'] = round(elapsed, 1)
    manifest_path = os.path.join(output_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SELF-HOSTING ENCODE COMPLETE")
    print(f"{'='*60}")
    print(f"  Files encoded: {len(manifest['files'])}")
    print(f"  Source bytes:  {manifest['total_source_bytes']:,}")
    print(f"  PNG bytes:     {manifest['total_png_bytes']:,}")
    print(f"  Ratio:         {manifest['total_png_bytes']/manifest['total_source_bytes']:.2f}x")
    print(f"  Time:          {elapsed:.1f}s")
    print(f"  Output:        {output_dir}/")
    
    return True


def main():
    # Determine output directory
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        output_dir = sys.argv[1]
    else:
        output_dir = os.path.join(os.path.dirname(__file__), 'self_host_pngs')
    
    # Determine which files to encode
    do_core_only = '--core' in sys.argv
    do_all = '--all' in sys.argv
    
    if do_core_only:
        files = list(SOURCE_FILES)
    elif do_all:
        base_dir = os.path.dirname(__file__)
        files = list(SOURCE_FILES)
        for f in OPTIONAL_FILES:
            if os.path.exists(os.path.join(base_dir, f)):
                files.append(f)
    else:
        # Default: core + any optional that exist
        base_dir = os.path.dirname(__file__)
        files = list(SOURCE_FILES)
        for f in OPTIONAL_FILES:
            if os.path.exists(os.path.join(base_dir, f)):
                files.append(f)
    
    ok = encode_source_files(output_dir, files)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

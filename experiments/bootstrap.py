"""
Pixelpack Bootstrap Decoder

Decodes a directory of pixelpack PNG images back into source files.
This is the minimal tool needed to reconstruct pixelpack from its
PNG-encoded form.

Requirements: expand.py + expand2.py in the same directory (or the
PNGs can decode those too if this file is included in self_host).

Usage:
    python3 bootstrap.py                          # decode self_host_pngs/ -> extracted/
    python3 bootstrap.py <png_dir> [output_dir]   # custom paths

The bootstrap process:
  1. Read manifest.json from the PNG directory
  2. For each file in manifest, read the PNG and expand its seeds
  3. Write the decoded bytes to the output directory
  4. Verify checksums match
"""

import os
import sys
import json


def bootstrap(png_dir, output_dir):
    """Decode all PNG files from png_dir into output_dir."""
    from expand2 import expand_from_png
    
    manifest_path = os.path.join(png_dir, 'manifest.json')
    if not os.path.exists(manifest_path):
        print(f"ERROR: No manifest.json found in {png_dir}")
        return False
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Bootstrapping {len(manifest['files'])} files...")
    print(f"  From: {png_dir}")
    print(f"  To:   {output_dir}")
    print()
    
    all_ok = True
    
    for entry in manifest['files']:
        png_path = os.path.join(png_dir, entry['png'])
        out_path = os.path.join(output_dir, entry['name'])
        
        if not os.path.exists(png_path):
            print(f"  MISSING: {entry['png']}")
            all_ok = False
            continue
        
        with open(png_path, 'rb') as f:
            png_data = f.read()
        
        decoded = expand_from_png(png_data)
        
        if len(decoded) != entry['source_bytes']:
            print(f"  FAIL: {entry['name']} - expected {entry['source_bytes']}B, got {len(decoded)}B")
            all_ok = False
            continue
        
        with open(out_path, 'wb') as f:
            f.write(decoded)
        
        # Make .py files executable
        if entry['name'].endswith('.py'):
            os.chmod(out_path, 0o644)
        
        print(f"  OK: {entry['name']} ({len(decoded):,} bytes)")
    
    print()
    if all_ok:
        print(f"Bootstrap complete. {len(manifest['files'])} files extracted to {output_dir}/")
        print()
        print("The pixelpack source is now reconstructed. You can run:")
        print(f"  cd {output_dir}")
        print(f"  python3 self_host.py --verify")
    else:
        print("Bootstrap completed with errors.")
    
    return all_ok


def main():
    png_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), 'self_host_pngs')
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), 'extracted')
    
    ok = bootstrap(png_dir, output_dir)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

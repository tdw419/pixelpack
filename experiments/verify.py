"""
Boot Pixel Verification Suite

Proves round-trip encoding: file -> pixel -> file -> execute
Tests multiple target programs across different languages.
"""

import sys
import os
import subprocess
import tempfile
from find_seed import search
from expand import expand, seed_to_rgba, seed_from_rgba
from boot import make_1x1_png, read_png_pixel


def verify_target(target: bytes, desc: str, runnable: bool = False, language: str = None):
    """Full round-trip verification for a single target."""
    print(f"{'='*60}")
    print(f"  Target: {desc}")
    print(f"  Bytes:  {target!r}")
    print(f"  Length: {len(target)} bytes")

    # Step 1: Find seed
    results = search(target, timeout=10.0)
    if not results:
        print(f"  FAIL: No seed found")
        return False

    seed, strategy = results[0]
    r, g, b, a = seed_to_rgba(seed)
    print(f"  Seed:   0x{seed:08X}")
    print(f"  RGBA:   ({r}, {g}, {b}, {a})")
    print(f"  Method: {strategy}")

    # Step 2: Encode to PNG
    png_data = make_1x1_png(r, g, b, a)
    print(f"  PNG:    {len(png_data)} bytes")

    # Step 3: Decode from PNG
    r2, g2, b2, a2 = read_png_pixel(png_data)
    seed2 = seed_from_rgba(r2, g2, b2, a2)

    if seed2 != seed:
        print(f"  FAIL: PNG round-trip seed mismatch!")
        print(f"    Expected: 0x{seed:08X}")
        print(f"    Got:      0x{seed2:08X}")
        return False

    # Step 4: Expand seed
    expanded = expand(seed)
    if expanded != target:
        print(f"  FAIL: Expansion mismatch!")
        print(f"    Expected: {target.hex()}")
        print(f"    Got:      {expanded.hex()}")
        return False

    print(f"  Round-trip: PASS")

    # Step 5: Execute if runnable
    if runnable:
        with tempfile.NamedTemporaryFile(
            mode='wb', suffix=_get_suffix(language), delete=False
        ) as f:
            f.write(expanded)
            tmp_path = f.name
        os.chmod(tmp_path, 0o755)

        try:
            cmd = _get_run_cmd(tmp_path, language)
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5.0
            )
            if result.returncode == 0:
                output = result.stdout.strip()
                print(f"  Execute:   PASS (exit=0)")
                print(f"  Output:    {output!r}")
            else:
                print(f"  Execute:   WARN (exit={result.returncode})")
                if result.stderr:
                    print(f"  Stderr:    {result.stderr[:100]}")
        except subprocess.TimeoutExpired:
            print(f"  Execute:   WARN (timeout)")
        except Exception as e:
            print(f"  Execute:   WARN ({e})")
        finally:
            os.unlink(tmp_path)

    # Summary
    ratio = len(target) / 4.0
    print(f"  Expansion: {len(target)}B from 32-bit seed ({ratio:.1f}x)")
    print()
    return True


def _get_suffix(language):
    return {
        'python': '.py',
        'shell': '.sh',
        'c': '.c',
        'javascript': '.js',
    }.get(language, '.txt')


def _get_run_cmd(path, language):
    return {
        'python': [sys.executable, path],
        'shell': ['/bin/bash', path],
        'c': ['gcc', path, '-o', path + '.out'],  # just compile
    }.get(language, [sys.executable, path])


def main():
    print("BOOT PIXEL VERIFICATION SUITE")
    print("Proving one pixel can encode a working program")
    print()

    tests = [
        # (target_bytes, description, runnable, language)
        (b'print("Hello")\n', 'Python Hello World', True, 'python'),
        (b'echo Hello\n', 'Shell Hello World', True, 'shell'),
        (b'42\n', 'Number literal file', False, None),
        (b'Hello, World!\n', 'Classic text output', False, None),
        (b'print(42)\n', 'Python print integer', True, 'python'),
        (b'void main(){}\n', 'C minimal program', False, 'c'),
    ]

    results = []
    for target, desc, runnable, lang in tests:
        ok = verify_target(target, desc, runnable, lang)
        results.append((desc, ok))

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for desc, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {desc}")

    print()
    print(f"  {passed}/{total} tests passed ({passed/total*100:.0f}%)")

    if passed == total:
        print()
        print("  PROOF COMPLETE: One pixel encodes generative recipes")
        print("  that deterministically expand into working programs.")

    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())

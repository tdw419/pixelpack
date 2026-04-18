"""
Boot Pixel Seed Searcher v4

Searches the 32-bit seed space for a value that expands to a target.
Strategy priority:
1. DICT_N (base 16-entry, up to 7 entries)
2. NIBBLE (7-byte fixed table)
3. DICTX5 (extended 32-entry, 5 entries)
4. BPE (4 x 7-bit byte-pair indices)
5. DICTX7 (sub-dict 16-31, 7 entries)
6. RLE
7. BYTEPACK
8. XOR_CHAIN
9. TEMPLATE
"""

import sys
import time
from expand import (
    expand, DICTIONARY, DICTIONARY_EXT, SUB_DICT, NIBBLE_TABLE,
    BPE_PAIR_TABLE,
    seed_from_rgba, seed_to_rgba
)


def search(target: bytes, timeout: float = 60.0) -> list:
    """Search for seeds that expand to the target bytes."""
    results = []
    start_time = time.time()
    tlen = len(target)

    print(f"Target: {target.hex()}")
    try:
        print(f"  ASCII: {target.decode('ascii')!r}")
    except UnicodeDecodeError:
        print(f"  Bytes: {repr(target)}")
    print(f"  Length: {tlen} bytes")
    print()

    # --- DICT_N (0x0-0x6): base dictionary decomposition ---
    for n in range(1, 8):
        if time.time() - start_time > timeout:
            break
        decomp = _decompose(target, n, DICTIONARY)
        if decomp is not None:
            seed = _pack_dict_seed(n, decomp)
            if _verify(seed, target):
                results.append((seed, f"DICT_{n}"))
                print(f"  FOUND via DICT_{n}: indices={decomp}")
                break

    if results:
        _print_results(results)
        return results

    # --- NIBBLE (0x7) ---
    r = _search_nibble(target)
    if r is not None:
        results.append((r, "NIBBLE"))
        print(f"  FOUND via NIBBLE")

    if results:
        _print_results(results)
        return results

    # --- DICTX5 (0x8): 5 entries from full DICTIONARY_EXT ---
    decomp = _decompose(target, 5, DICTIONARY_EXT)
    if decomp is not None:
        params = 0
        for i, idx in enumerate(decomp):
            params |= (idx & 0x1F) << (5 * i)
        seed = 0x80000000 | params
        if _verify(seed, target):
            results.append((seed, "DICTX5"))
            print(f"  FOUND via DICTX5: indices={decomp}")

    if results:
        _print_results(results)
        return results

    # --- BPE (0x9): 4 x 7-bit byte-pair indices ---
    r = _search_bpe(target)
    if r is not None:
        results.append((r, "BPE"))
        print(f"  FOUND via BPE")

    if results:
        _print_results(results)
        return results

    # --- DICTX7 (0xA): 7 entries from SUB_DICT ---
    decomp = _decompose_sub(target, 7)
    if decomp is not None:
        params = 0
        for i, idx in enumerate(decomp):
            params |= (idx & 0xF) << (4 * i)
        seed = 0xA0000000 | params
        if _verify(seed, target):
            results.append((seed, "DICTX7"))
            print(f"  FOUND via DICTX7: indices={decomp}")

    if results:
        _print_results(results)
        return results

    # --- RLE (0xB) ---
    r = _search_rle(target)
    if r is not None:
        results.append((r, "RLE"))
        print(f"  FOUND via RLE")

    if results:
        _print_results(results)
        return results

    # --- BYTEPACK (0xE) ---
    r = _search_bytepack(target)
    if r is not None:
        results.append((r, "BYTEPACK"))
        print(f"  FOUND via BYTEPACK")

    if results:
        _print_results(results)
        return results

    # --- XOR_CHAIN (0xC) ---
    r = _search_xor_chain(target, start_time, timeout)
    if r is not None:
        results.append((r, "XOR_CHAIN"))
        print(f"  FOUND via XOR_CHAIN")

    if results:
        _print_results(results)
        return results

    # --- TEMPLATE (0xF) ---
    r = _search_template(target)
    if r is not None:
        results.append((r, "TEMPLATE"))
        print(f"  FOUND via TEMPLATE")

    if results:
        _print_results(results)
        return results

    elapsed = time.time() - start_time
    print(f"\nNo exact match found after {elapsed:.1f}s")
    print("  Target needs dictionary update or multi-pixel encoding.")
    return results


# === Core helpers ===

def _verify(seed, target):
    try:
        return expand(seed, len(target) + 1) == target
    except Exception:
        return False


def _decompose(target, n_entries, dictionary):
    """Find decomposition into exactly n_entries consecutive dictionary entries."""
    if n_entries == 0:
        return [] if len(target) == 0 else None
    return _decomp_rec(target, 0, n_entries, dictionary)


def _decomp_rec(target, pos, remaining, dictionary):
    if remaining == 0:
        return [] if pos == len(target) else None
    if pos >= len(target):
        return None

    # Pruning: min/max bytes remaining entries can produce
    lengths = [len(e) for e in dictionary]
    min_rem = remaining * min(lengths)
    max_rem = remaining * max(lengths)
    bytes_left = len(target) - pos
    if bytes_left < min_rem or bytes_left > max_rem:
        return None

    for i, entry in enumerate(dictionary):
        elen = len(entry)
        if pos + elen <= len(target) and target[pos:pos + elen] == entry:
            rest = _decomp_rec(target, pos + elen, remaining - 1, dictionary)
            if rest is not None:
                return [i] + rest
    return None


def _decompose_sub(target, n_entries):
    """Decompose target into n_entries from SUB_DICT (4-bit indices 0-15)."""
    return _decompose(target, n_entries, SUB_DICT)


def _pack_dict_seed(n, indices):
    """Build 32-bit seed for DICT_N strategy (4-bit indices)."""
    params = 0
    for i, idx in enumerate(indices):
        params |= (idx & 0xF) << (4 * i)
    return ((n - 1) << 28) | params


# === Strategy-specific searchers ===

def _search_nibble(target):
    if len(target) != 7:
        return None
    byte_to_nibble = {}
    for i, b in enumerate(NIBBLE_TABLE):
        byte_to_nibble[b] = i
    nibbles = []
    for b in target:
        if b not in byte_to_nibble:
            return None
        nibbles.append(byte_to_nibble[b])
    params = 0
    for i, nib in enumerate(nibbles):
        params |= (nib & 0xF) << (4 * i)
    seed = 0x70000000 | params
    return seed if _verify(seed, target) else None


def _search_bpe(target):
    """Try to encode target as BPE byte-pair sequence (2, 4, 6, or 8 bytes)."""
    tlen = len(target)
    if tlen == 0 or tlen > 8 or tlen % 2 != 0:
        return None

    # Build reverse lookup
    pair_to_idx = {}
    for i, pair in enumerate(BPE_PAIR_TABLE):
        if i > 0 and pair:
            pair_to_idx[pair] = i

    n_pairs = tlen // 2
    indices = []
    for i in range(n_pairs):
        pair = target[i*2:i*2+2]
        idx = pair_to_idx.get(pair)
        if idx is None:
            return None
        indices.append(idx)

    # Pack: 4 x 7-bit indices, unused slots = 0 (terminator)
    params = 0
    for i in range(4):
        if i < n_pairs:
            params |= (indices[i] & 0x7F) << (7 * i)
        # else: 0 = terminator

    seed = 0x90000000 | params
    return seed if _verify(seed, target) else None


def _search_rle(target):
    tlen = len(target)
    for repeats in range(1, 17):
        if tlen % repeats != 0:
            continue
        pattern_len = tlen // repeats
        for count_a in range(1, min(17, pattern_len)):
            count_b = pattern_len - count_a
            if count_b < 1 or count_b > 16:
                continue
            byte_a = target[0]
            byte_b = target[count_a]
            valid = True
            for r in range(repeats):
                base = r * pattern_len
                for j in range(count_a):
                    if target[base + j] != byte_a:
                        valid = False; break
                if not valid:
                    break
                for j in range(count_b):
                    if target[base + count_a + j] != byte_b:
                        valid = False; break
                if not valid:
                    break
            if valid:
                params = (byte_a & 0xFF) | ((byte_b & 0xFF) << 8) | \
                         (((count_a - 1) & 0xF) << 16) | \
                         (((count_b - 1) & 0xF) << 20) | \
                         (((repeats - 1) & 0xF) << 24)
                seed = 0xB0000000 | params
                if _verify(seed, target):
                    return seed
    return None


def _search_xor_chain(target, start_time, timeout):
    tlen = len(target)
    if tlen > 16 or tlen == 0:
        return None
    start = target[0]
    for key in range(256):
        if time.time() - start_time > timeout:
            return None
        for mask in [0xFF] + [m for m in range(1, 256) if m != 0xFF]:
            val = start
            match = True
            for i in range(tlen):
                if val != target[i]:
                    match = False; break
                val = ((val ^ key) & mask)
                if val == 0:
                    val = key
            if match:
                count_bits = (tlen - 1) & 0xF
                params = (start & 0xFF) | ((key & 0xFF) << 8) | \
                         ((mask & 0xFF) << 16) | (count_bits << 24)
                seed = 0xC0000000 | params
                if _verify(seed, target):
                    return seed
    return None


def _search_bytepack(target):
    """Try all BYTEPACK sub-modes analytically."""
    tlen = len(target)

    # Mode 0: 3 raw bytes + optional repeat
    if 3 <= tlen <= 18:
        for extra in range(0, min(16, tlen - 3 + 1)):
            if tlen == 3 + extra:
                b0, b1, b2 = target[0], target[1], target[2]
                if extra == 0 or all(target[3+i] == b0 for i in range(extra)):
                    data = b0 | (b1 << 8) | (b2 << 16) | (extra << 24)
                    seed = 0xE0000000 | (0 << 0) | (data << 3)
                    if _verify(seed, target):
                        return seed

    # Mode 1: XOR delta
    if 3 <= tlen <= 18 and tlen >= 3:
        base = target[0]
        d1 = target[0] ^ target[1]
        d2 = (target[0] ^ target[1]) ^ target[2] if tlen > 2 else 0
        check = [base, base ^ d1, (base ^ d1) ^ d2]
        if check[:min(tlen,3)] == list(target[:min(tlen,3)]):
            for extra in range(0, min(16, tlen - 3 + 1)):
                if tlen == 3 + extra:
                    if extra == 0 or all(target[3+i] == check[-1] for i in range(extra)):
                        data = base | (d1 << 8) | (d2 << 16) | (extra << 24)
                        seed = 0xE0000000 | (1 << 0) | (data << 3)
                        if _verify(seed, target):
                            return seed

    # Mode 2: ADD delta (3-4 bytes)
    if 3 <= tlen <= 4:
        base = target[0]
        d1 = (target[1] - base) & 0xFF
        d2 = (target[2] - target[1]) & 0xFF
        d3 = (target[3] - target[2]) & 0xF if tlen > 3 else 0
        data = base | (d1 << 8) | (d2 << 16) | (d3 << 24)
        seed = 0xE0000000 | (2 << 0) | (data << 3)
        if _verify(seed, target):
            return seed

    # Mode 3: 4 nibbles with shared high nibble
    if tlen == 4:
        hi_nibble = target[0] >> 4
        if all((b >> 4) == hi_nibble for b in target):
            data = hi_nibble | ((target[0] & 0xF) << 4) | ((target[1] & 0xF) << 8) | \
                   ((target[2] & 0xF) << 12) | ((target[3] & 0xF) << 16)
            seed = 0xE0000000 | (3 << 0) | (data << 3)
            if _verify(seed, target):
                return seed

    # Mode 4: 4 bytes, 7 bits each
    if tlen == 4 and all(b <= 127 for b in target):
        data = (target[0] & 0x7F) | ((target[1] & 0x7F) << 7) | \
               ((target[2] & 0x7F) << 14) | ((target[3] & 0x7F) << 21)
        seed = 0xE0000000 | (4 << 0) | (data << 3)
        if _verify(seed, target):
            return seed

    # Mode 5: Shared base + 4 nibble offsets
    if tlen == 4:
        for base in range(256):
            offsets = [(b - base) & 0xFF for b in target]
            if all(0 <= o <= 15 for o in offsets):
                data = base | (offsets[0] << 8) | (offsets[1] << 12) | \
                       (offsets[2] << 16) | (offsets[3] << 20)
                seed = 0xE0000000 | (5 << 0) | (data << 3)
                if _verify(seed, target):
                    return seed

    # Mode 6: 5 bytes via lowercase+digit 5-bit table
    if tlen == 5:
        table = 'abcdefghijklmnopqrstuvwxyz012345'
        try:
            indices = [table.index(chr(b)) for b in target]
            data = sum(idx << (5 * i) for i, idx in enumerate(indices))
            seed = 0xE0000000 | (6 << 0) | (data << 3)
            if _verify(seed, target):
                return seed
        except (ValueError, OverflowError):
            pass

    # Mode 7: 5 bytes via uppercase+symbol 5-bit table
    if tlen == 5:
        table = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ !,.\n('
        try:
            indices = [table.index(chr(b)) for b in target]
            data = sum(idx << (5 * i) for i, idx in enumerate(indices))
            seed = 0xE0000000 | (7 << 0) | (data << 3)
            if _verify(seed, target):
                return seed
        except (ValueError, OverflowError):
            pass

    return None


def _search_template(target):
    templates = [
        b'Hello, World!\n', b'print("hi")\n', b'echo hello\n',
        b'int main(){}\n', b'mov r0, #1\n', b'AAAA', b'BBBB', b'CCCC',
        b'ld a, 0\n', b'push 42\n', b'x = 1\n', b'a = b\n',
        b'fn f()\n', b'pub fn\n', b'val x\n', b'let x\n',
    ]
    tlen = len(target)
    for idx, template in enumerate(templates):
        expected_len = len(template) + 2
        if tlen != expected_len:
            continue
        key = (target[0] - template[0]) & 0xFF
        if all((template[i] + key) & 0xFF == target[i] for i in range(len(template))):
            extra1 = target[len(template)]
            extra2 = target[len(template) + 1]
            params = (idx & 0xF) | ((key & 0xFF) << 4) | \
                     ((extra1 & 0xFF) << 12) | ((extra2 & 0xFF) << 20)
            seed = 0xF0000000 | params
            if _verify(seed, target):
                return seed
    return None


def _print_results(results):
    print(f"\n{'='*50}")
    print(f"Found {len(results)} matching seed(s):")
    for seed, name in results:
        r, g, b, a = seed_to_rgba(seed)
        output = expand(seed)
        print(f"  Seed: 0x{seed:08X}  Strategy: {name}")
        print(f"  RGBA: ({r}, {g}, {b}, {a})")
        print(f"  Output ({len(output)} bytes): {output.hex()}")
        try:
            print(f"  Text: {output.decode('ascii')!r}")
        except UnicodeDecodeError:
            pass
    print(f"{'='*50}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Boot Pixel Seed Searcher v4")
        print("Usage:")
        print("  python3 find_seed.py --text \"print('Hello')\" ")
        print("  python3 find_seed.py --hex 48656C6C6F")
        print("  python3 find_seed.py --demo")
        sys.exit(1)

    if sys.argv[1] == '--demo':
        for target in [b'print("Hello")\n', b'echo Hello\n', b'42\n']:
            search(target)
            print()
    elif sys.argv[1] == '--text':
        results = search(sys.argv[2].encode('utf-8'))
        if not results:
            sys.exit(1)
    elif sys.argv[1] == '--hex':
        results = search(bytes.fromhex(sys.argv[2]))
        if not results:
            sys.exit(1)

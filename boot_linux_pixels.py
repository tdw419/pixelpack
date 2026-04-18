#!/usr/bin/env python3
"""Boot Linux from pixels.

Encode a RISC-V Linux kernel + initramfs as .rts.png pixel images,
then prepare for boot via the Geometry OS RISC-V interpreter.

Usage:
    python3 boot_linux_pixels.py --encode <vmlinux> <initramfs.cpio.gz> <output_dir>
    python3 boot_linux_pixels.py --verify <output_dir>
    python3 boot_linux_pixels.py --info <kernel.rts.png>

The output directory will contain:
    - kernel.rts.png   -- the RV32 Linux kernel as pixels
    - initramfs.rts.png -- the root filesystem as pixels
    - boot_meta.json   -- metadata for the Geometry OS boot loader
"""
import os, sys, json, hashlib, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_to_rts import encode, decode


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def encode_boot(vmlinux_path, initramfs_path, output_dir):
    """Encode kernel + initramfs as pixel images."""
    os.makedirs(output_dir, exist_ok=True)

    kernel_png = os.path.join(output_dir, "kernel.rts.png")
    initrd_png = os.path.join(output_dir, "initramfs.rts.png")
    meta_path = os.path.join(output_dir, "boot_meta.json")

    # Encode kernel
    print(f"Encoding kernel: {vmlinux_path}")
    kernel_size = os.path.getsize(vmlinux_path)
    kernel_hash = sha256_file(vmlinux_path)
    encode(vmlinux_path, kernel_png, mode="raw")
    kernel_png_size = os.path.getsize(kernel_png)

    # Encode initramfs
    print(f"\nEncoding initramfs: {initramfs_path}")
    initrd_size = os.path.getsize(initramfs_path)
    initrd_hash = sha256_file(initramfs_path)
    encode(initramfs_path, initrd_png, mode="raw")
    initrd_png_size = os.path.getsize(initrd_png)

    # Write metadata
    meta = {
        "architecture": "riscv32",
        "abi": "ilp32",
        "kernel": {
            "source": os.path.basename(vmlinux_path),
            "original_size": kernel_size,
            "pixel_size": kernel_png_size,
            "pixel_ratio": round(kernel_png_size / kernel_size, 3),
            "sha256": kernel_hash,
        },
        "initramfs": {
            "source": os.path.basename(initramfs_path),
            "original_size": initrd_size,
            "pixel_size": initrd_png_size,
            "pixel_ratio": round(initrd_png_size / initrd_size, 3),
            "sha256": initrd_hash,
        },
        "boot_protocol": "sbi",
        "entry_point": "0xC0000000",
        "format": "pixelpack.rts.v1",
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nBoot images ready in {output_dir}/")
    print(f"  kernel.rts.png    : {kernel_png_size:,} bytes ({kernel_png_size/kernel_size*100:.1f}% of {kernel_size:,})")
    print(f"  initramfs.rts.png : {initrd_png_size:,} bytes ({initrd_png_size/initrd_size*100:.1f}% of {initrd_size:,})")
    print(f"  boot_meta.json    : metadata")
    print(f"\nTotal pixel payload: {kernel_png_size + initrd_png_size:,} bytes")
    print(f"Original binaries  : {kernel_size + initrd_size:,} bytes")
    print(f"Overall ratio      : {(kernel_png_size + initrd_png_size)/(kernel_size + initrd_size)*100:.1f}%")

    return meta


def verify_boot(output_dir):
    """Verify pixel images decode back to byte-perfect originals."""
    meta_path = os.path.join(output_dir, "boot_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    all_ok = True

    # Verify kernel
    kernel_png = os.path.join(output_dir, "kernel.rts.png")
    kernel_decoded = "/tmp/_verify_kernel"
    print("Verifying kernel.rts.png ...")
    decode(kernel_png, kernel_decoded)
    dec_hash = sha256_file(kernel_decoded)
    ok = dec_hash == meta["kernel"]["sha256"]
    print(f"  SHA256: {'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        all_ok = False
    os.unlink(kernel_decoded)

    # Verify initramfs
    initrd_png = os.path.join(output_dir, "initramfs.rts.png")
    initrd_decoded = "/tmp/_verify_initrd"
    print("Verifying initramfs.rts.png ...")
    decode(initrd_png, initrd_decoded)
    dec_hash = sha256_file(initrd_decoded)
    ok = dec_hash == meta["initramfs"]["sha256"]
    print(f"  SHA256: {'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        all_ok = False
    os.unlink(initrd_decoded)

    print(f"\n{'PASS' if all_ok else 'FAIL'}: Boot images verified")
    return all_ok


def show_info(png_path):
    """Show info about a .rts.png file."""
    from PIL import Image
    img = Image.open(png_path)
    info = img.info or {}

    print(f"File: {png_path}")
    print(f"Size: {img.size[0]}x{img.size[1]} pixels")
    print(f"Mode: {img.mode}")
    total = img.size[0] * img.size[1]
    data_size = int(info.get("data_size", total * 4))
    print(f"Data size: {data_size:,} bytes ({data_size/1024/1024:.1f} MB)")
    print(f"File size: {os.path.getsize(png_path):,} bytes")
    if "sha256" in info:
        print(f"SHA256: {info['sha256']}")
    if "source" in info:
        print(f"Source: {info['source']}")
    if "mode" in info:
        print(f"Encode mode: {info['mode']}")


def main():
    p = argparse.ArgumentParser(description="Boot Linux from pixels")
    sub = p.add_subparsers(dest="cmd")

    enc = sub.add_parser("encode", help="Encode kernel + initramfs as pixels")
    enc.add_argument("vmlinux", help="Path to RV32 vmlinux ELF")
    enc.add_argument("initramfs", help="Path to initramfs.cpio.gz")
    enc.add_argument("output_dir", help="Output directory for .rts.png files")

    ver = sub.add_parser("verify", help="Verify pixel boot images")
    ver.add_argument("output_dir", help="Directory with .rts.png files")

    inf = sub.add_parser("info", help="Show info about a .rts.png file")
    inf.add_argument("png_path", help="Path to .rts.png")

    args = p.parse_args()

    if args.cmd == "encode":
        encode_boot(args.vmlinux, args.initramfs, args.output_dir)
    elif args.cmd == "verify":
        ok = verify_boot(args.output_dir)
        sys.exit(0 if ok else 1)
    elif args.cmd == "info":
        show_info(args.png_path)
    else:
        p.print_help()


if __name__ == "__main__":
    main()

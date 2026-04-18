#!/usr/bin/env python3
"""Pixel Model Server -- load a model from .rts.png and serve it for inference.

Pipeline:
  1. Download/locate an open-source GGUF model
  2. Convert to .rts.png (if not already done)
  3. Load .rts.png → decode to GGUF → llama-cpp-python → inference
  4. Use the model as a coding assistant for Geometry OS

Usage:
  python3 pixel_model_server.py --convert <model.gguf> <output.rts.png>
  python3 pixel_model_server.py --serve <model.rts.png>
  python3 pixel_model_server.py --serve <model.rts.png> --prompt "write a hello world in Rust"
  python3 pixel_model_server.py --geo <model.rts.png>  # Geometry OS mode
"""
import os, sys, argparse, tempfile, subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def convert_to_pixel(gguf_path, output_png):
    """Convert GGUF → .rts.png."""
    from gguf_to_rts import encode
    print(f"Converting {gguf_path} → {output_png} ...")
    encode(gguf_path, output_png, mode="raw")
    
    original_size = os.path.getsize(gguf_path)
    pixel_size = os.path.getsize(output_png)
    print(f"\n  Original: {original_size:,} bytes")
    print(f"  Pixel:    {pixel_size:,} bytes")
    print(f"  Ratio:    {pixel_size/original_size*100:.1f}%")
    print(f"\nThe model is now a viewable PNG image.")


def decode_to_gguf(rts_png_path):
    """Decode .rts.png → temp GGUF file, return path."""
    from gguf_to_rts import decode
    
    tmp = tempfile.mktemp(suffix=".gguf", prefix="pixel_model_",
                          dir=os.path.dirname(os.path.abspath(rts_png_path)))
    print(f"Decoding pixel model → {tmp}")
    success = decode(rts_png_path, tmp)
    if not success:
        print("ERROR: SHA256 mismatch! Model may be corrupted.")
        sys.exit(1)
    
    fsize = os.path.getsize(tmp)
    print(f"Decoded: {fsize:,} bytes ({fsize/1024/1024:.1f} MB)")
    return tmp


def serve_model(gguf_path, prompt=None, geo_mode=False, interactive=True):
    """Load model with llama-cpp-python and run inference."""
    try:
        from llama_cpp import Llama
    except ImportError:
        print("ERROR: llama-cpp-python not installed.")
        print("  pip install llama-cpp-python")
        sys.exit(1)
    
    print(f"\nLoading model from {gguf_path} ...")
    print("(this may take a moment for large models)")
    
    llm = Llama(
        model_path=gguf_path,
        n_ctx=4096,
        n_gpu_layers=-1,  # offload all to GPU
        verbose=False,
    )
    print("Model loaded.\n")
    
    # System prompts
    geo_system = """You are a systems programmer working on Geometry OS, a visual computing operating system.
Geometry OS uses 100 assembler mnemonics, multi-process execution, a virtual filesystem (VFS),
and renders everything as pixels. The OS runs on a RISC-V VM and displays on an infinite pixel map.

Key concepts:
- Pixels are the fundamental unit of compute and display
- The assembler has instructions for pixel manipulation, memory, branching, and I/O
- Programs are compiled from .glyph assembly files
- The VFS provides /dev/display, /dev/keyboard, /proc filesystems
- The GPU execution path uses compute shaders via WebGPU/WGSL

When writing code, be precise about instruction names, register usage, and pixel coordinates.
Output clean, working code with brief explanations."""

    default_system = "You are a helpful coding assistant. Be concise and write working code."
    
    system_prompt = geo_system if geo_mode else default_system
    
    if prompt:
        # Single prompt mode
        response = chat(llm, system_prompt, prompt)
        print(response)
        return
    
    if interactive:
        interactive_chat(llm, system_prompt)


def chat(llm, system_prompt, user_message, max_tokens=1024):
    """Run a single chat completion."""
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
    )
    return response["choices"][0]["message"]["content"]


def interactive_chat(llm, system_prompt):
    """Run interactive chat loop."""
    messages = [{"role": "system", "content": system_prompt}]
    
    print("=" * 60)
    print("Pixel Model Server -- Interactive Chat")
    print("Type your messages. 'quit' to exit, 'clear' to reset.")
    print("=" * 60)
    print()
    
    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.lower() == "clear":
            messages = [{"role": "system", "content": system_prompt}]
            print("Context cleared.\n")
            continue
        
        messages.append({"role": "user", "content": user_input})
        
        print("Model> ", end="", flush=True)
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            stream=True,
        )
        
        full_response = ""
        for chunk in response:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                print(content, end="", flush=True)
                full_response += content
        
        print("\n")
        messages.append({"role": "assistant", "content": full_response})


def main():
    parser = argparse.ArgumentParser(description="Pixel Model Server")
    parser.add_argument("--convert", nargs=2, metavar=("GGUF", "PNG"),
                       help="Convert GGUF to .rts.png")
    parser.add_argument("--serve", metavar="PNG",
                       help="Serve model from .rts.png")
    parser.add_argument("--prompt", type=str,
                       help="Single prompt (non-interactive)")
    parser.add_argument("--geo", action="store_true",
                       help="Geometry OS coding mode")
    parser.add_argument("--no-interactive", action="store_true",
                       help="Don't start interactive chat")
    
    args = parser.parse_args()
    
    if args.convert:
        convert_to_pixel(args.convert[0], args.convert[1])
    
    elif args.serve:
        # Decode .rts.png → GGUF → load
        gguf_path = decode_to_gguf(args.serve)
        serve_model(
            gguf_path,
            prompt=args.prompt,
            geo_mode=args.geo,
            interactive=not args.no_interactive,
        )
        # Cleanup temp file
        try:
            os.unlink(gguf_path)
        except:
            pass
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

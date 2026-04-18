#!/usr/bin/env python3
"""Train a model whose weights live as pixels in a .rts.png file.

Pipeline:
  1. Load .rts.png → decode Hilbert → raw bytes
  2. Parse bytes as GGUF tensor layout → F32 weight tensors
  3. Wrap tensors as PyTorch parameters
  4. Run training loop (forward, loss, backward, update)
  5. Write updated weights back → encode Hilbert → save .rts.png

The checkpoint IS the image. Watch weights evolve visually.
"""
import struct, math, os, sys, json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_to_rts import (
    parse_gguf, hilbert_xy_array, delta_encode, delta_decode
)

# ── Pixel ↔ Weight Bridge ─────────────────────────────────────

class PixelWeightStore:
    """Manages the mapping between .rts.png pixels and model weights.
    
    The .rts.png stores the entire GGUF binary as RGBA pixels.
    We decode → extract tensor slices → create PyTorch params.
    After training, we pack updated params back → encode → save PNG.
    """
    
    def __init__(self, rts_png_path):
        self.rts_path = rts_png_path
        self.gguf_path = None
        self.raw_data = None
        self.tensors = None
        self.data_start = None
        self.metadata = None
        
    def load_from_gguf(self, gguf_path):
        """Load GGUF, parse tensors, store raw binary."""
        self.gguf_path = gguf_path
        self.metadata, self.tensors, self.raw_data, self.data_start = \
            parse_gguf(gguf_path)
        print(f"Loaded GGUF: {len(self.raw_data):,} bytes, "
              f"{len(self.tensors)} tensors")
        return self.raw_data
    
    def load_from_rts(self, rts_png_path):
        """Load from .rts.png, decode to raw binary."""
        from gguf_to_rts import decode
        # Decode to temp file
        tmp_path = "/tmp/_rts_decoded.gguf"
        decode(rts_png_path, tmp_path)
        return self.load_from_gguf(tmp_path)
    
    def extract_tensors(self):
        """Extract all tensors as numpy arrays with their shapes."""
        result = {}
        for t in self.tensors:
            n_elements = 1
            for d in t["dims"]:
                n_elements *= d
            toff = self.data_start + t["offset"]
            
            if t["type_id"] == 0:  # F32
                raw = self.raw_data[toff:toff + n_elements * 4]
                arr = np.frombuffer(raw, dtype=np.float32).copy()
            elif t["type_id"] == 1:  # F16
                raw = self.raw_data[toff:toff + n_elements * 2]
                arr = np.frombuffer(raw, dtype=np.float16).copy().astype(np.float32)
            else:
                # For quantized tensors, just grab the raw bytes
                # (training would need dequantization - skip for now)
                bpe = {2: 0.5625, 3: 0.625, 6: 0.6875, 7: 0.75, 8: 1.0625,
                       9: 1.125, 10: 0.3125, 11: 0.4375, 12: 0.5625,
                       13: 0.6875, 14: 0.8125, 15: 1.0625}
                tsize = int(n_elements * bpe.get(t["type_id"], 1.0))
                raw = self.raw_data[toff:toff + tsize]
                arr = np.frombuffer(raw, dtype=np.uint8).copy().astype(np.float32) / 255.0
            
            result[t["name"]] = {
                "data": arr,
                "shape": t["dims"],
                "type_id": t["type_id"],
                "offset": t["offset"],
                "size": n_elements,
            }
        return result
    
    def inject_tensors(self, tensor_dict):
        """Write updated tensor data back into raw_data."""
        for t in self.tensors:
            if t["name"] not in tensor_dict:
                continue
            updated = tensor_dict[t["name"]]
            toff = self.data_start + t["offset"]
            
            if t["type_id"] == 0:  # F32
                arr = updated.astype(np.float32)
                self.raw_data = (self.raw_data[:toff] + 
                                arr.tobytes() + 
                                self.raw_data[toff + len(arr.tobytes()):])
            elif t["type_id"] == 1:  # F16
                arr = updated.astype(np.float16)
                self.raw_data = (self.raw_data[:toff] + 
                                arr.tobytes() + 
                                self.raw_data[toff + len(arr.tobytes()):])
        return self.raw_data
    
    def save_rts_png(self, output_path, mode="raw"):
        """Encode current raw_data as .rts.png."""
        from gguf_to_rts import encode
        # Write temp GGUF, then encode
        tmp_gguf = "/tmp/_rts_encode_input.gguf"
        with open(tmp_gguf, "wb") as f:
            f.write(self.raw_data)
        encode(tmp_gguf, output_path, mode=mode)


# ── Simple Model from GGUF Tensors ────────────────────────────

class VectorOSNet(nn.Module):
    """Build a simple model from VectorOS GGUF tensors.
    
    The VectorOS model has:
    - kernel.farsight.w1: [256, 64] (input projection)
    - kernel.farsight.w2: [64, 1]   (output projection)
    - kernel.skill.*:    [64, 64]   (skill modules)
    - visual.encoder_*:  various    (vision encoder)
    - visual.decoder_*:  various    (vision decoder)
    """
    
    def __init__(self, tensor_data):
        super().__init__()
        self.params = nn.ParameterDict()
        
        for name, info in tensor_data.items():
            # Convert numpy to torch parameter
            t = torch.from_numpy(info["data"].copy())
            shape = info["shape"]
            
            # Reshape to proper dimensions
            if len(shape) == 2:
                t = t.view(shape[0], shape[1])
            elif len(shape) == 1:
                t = t.view(shape[0])
            elif len(shape) == 4:
                t = t.view(shape[0], shape[1], shape[2], shape[3])
            
            self.params[name.replace(".", "_")] = nn.Parameter(t)
        
        self.tensor_info = tensor_data
    
    def forward(self, x):
        """Simple forward pass: farsight pathway.
        w1 projects 256→64, w2 projects 64→1.
        """
        w1 = self.params["kernel_farsight_w1"]  # [256, 64]
        w2 = self.params["kernel_farsight_w2"]  # [64, 1]
        
        # x: [batch, 256]
        h = torch.matmul(x, w1)  # [batch, 64]
        h = torch.relu(h)
        out = torch.matmul(h, w2)  # [batch, 1]
        return out
    
    def get_updated_numpy(self):
        """Return dict of name→numpy with updated weights."""
        result = {}
        for name, info in self.tensor_info.items():
            param = self.params[name.replace(".", "_")]
            result[name] = param.detach().cpu().numpy().flatten()
        return result


# ── Training Demo ──────────────────────────────────────────────

def train_demo(gguf_path, output_png, steps=100, lr=0.01):
    """Demo: load GGUF, build model, train, save as .rts.png."""
    
    # 1. Load from GGUF
    store = PixelWeightStore(gguf_path)
    store.load_from_gguf(gguf_path)
    
    # 2. Extract tensors
    tensor_data = store.extract_tensors()
    
    # 3. Build PyTorch model
    model = VectorOSNet(tensor_data)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params:,} parameters")
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Show initial weights
    w1 = model.params["kernel_farsight_w1"]
    print(f"\nInitial w1: shape={w1.shape}, "
          f"range=[{w1.min():.4f}, {w1.max():.4f}], mean={w1.mean():.4f}")
    
    # 4. Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    print(f"\nTraining {steps} steps...")
    losses = []
    
    for step in range(steps):
        # Random input (small scale to avoid explosion)
        x = torch.randn(8, 256) * 0.1
        target = torch.randn(8, 1) * 0.1
        
        # Forward
        out = model(x)
        loss = loss_fn(out, target)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses.append(loss.item())
        if (step + 1) % 20 == 0:
            print(f"  Step {step+1}: loss={loss.item():.4f}")
    
    # 5. Show weight changes
    w1_new = model.params["kernel_farsight_w1"]
    w1_orig = torch.from_numpy(tensor_data["kernel.farsight.w1"]["data"].copy())
    w1_orig = w1_orig.view(256, 64)
    diff = (w1_new - w1_orig).abs()
    print(f"\nWeight changes in w1:")
    print(f"  Max delta: {diff.max():.6f}")
    print(f"  Mean delta: {diff.mean():.6f}")
    print(f"  % changed > 0.001: {(diff > 0.001).float().mean()*100:.1f}%")
    
    # 6. Write updated weights back to raw_data
    updated = model.get_updated_numpy()
    store.inject_tensors(updated)
    
    # 7. Save as .rts.png
    print(f"\nSaving trained model as {output_png}...")
    store.save_rts_png(output_png)
    
    # 8. Verify round-trip
    print("\nVerifying round-trip...")
    store2 = PixelWeightStore(output_png)
    store2.load_from_rts(output_png)
    td2 = store2.extract_tensors()
    
    # Check a tensor
    orig = tensor_data["kernel.farsight.w1"]["data"]
    reloaded = td2["kernel.farsight.w1"]["data"]
    if np.allclose(orig, updated["kernel.farsight.w1"], atol=1e-6):
        print("  Updated weights preserved: YES")
    if np.allclose(reloaded, updated["kernel.farsight.w1"], atol=1e-6):
        print("  Round-trip verified: YES")
    
    return model, losses


if __name__ == "__main__":
    gguf_path = "/home/jericho/.vectoros/lnx/vectoros_kernel_v1.gguf"
    output_png = "/tmp/vectoros_trained.rts.png"
    
    steps = 100
    lr = 0.01
    
    if len(sys.argv) > 1:
        gguf_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_png = sys.argv[2]
    if "--steps" in sys.argv:
        steps = int(sys.argv[sys.argv.index("--steps") + 1])
    if "--lr" in sys.argv:
        lr = float(sys.argv[sys.argv.index("--lr") + 1])
    
    model, losses = train_demo(gguf_path, output_png, steps=steps, lr=lr)
    
    print(f"\nDone. Trained model saved as viewable PNG: {output_png}")
    print(f"You can open it in any image viewer to see the weight structure.")

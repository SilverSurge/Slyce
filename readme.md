# 🧠 Slyce: Dynamic Slice-Based PyTorch Models

**Slyce** is a dynamic memory-aware architecture for PyTorch that lets you load, unload, and cache model components ("slices") across GPU, CPU, and disk—enabling large models to run efficiently even on limited hardware.

## ✨ Features

- **Slice-Based Execution**: Only one slice is active at a time to conserve memory.
- **GPU Strategy Options**:
  - `'none'`: Run on CPU only.
  - `'active_only'`: Only keep the currently active slice on GPU.
  - `'limit_gpu'`: Keep as many slices as fit within a GPU memory budget.
- **Multi-Level Caching**: Automatically moves slices from GPU → CPU → disk.
- **Memory Tracking**: Uses PyTorch and `psutil` to monitor GPU/CPU memory usage.
- **Smart Logging**: Logs slice activation, memory usage, and caching behavior to both file and console.
- **Modular Design**: Each slice is built from a constructor and config for full flexibility.


## 📦 Installation

```bash
git clone https://github.com/yourusername/slyce.git
cd slyce
pip install -r requirements.txt
```

> ⚠️ Make sure your environment supports PyTorch and `psutil`.

---

## 🧪 Quick Start Example (SUBJECT TO CHANGES)

```python
from slyce import SlyceModel

# Define your slices
slices = [
    {"name": "encoder", "constructor": Encoder, "config": encoder_cfg},
    {"name": "decoder", "constructor": Decoder, "config": decoder_cfg}
]

# Initialize Slyce model
model = SlyceModel(
    slices=slices,
    gpu_strategy="limit_gpu",
    max_gpu_mem_mb=6000,
    disk_cache_path="./slyce_cache"
)

# Use the encoder slice
output = model.forward(input_tensor, active_slice="encoder")

# Switch to decoder
model.set_active_slice("decoder")
output = model.forward(input_tensor)
```

---

## 📂 Project Structure (STILL IN Development Stages)

```
slyce/
├── __init__.py
├── core.py         # Core logic of SlyceModel
├── memory.py       # GPU/CPU memory tracking
├── cache.py        # Disk and CPU caching
├── utils.py        # Logging and file utilities
```

---

## 🧠 Design Goals

- Efficient use of GPU memory without sacrificing modularity.
- Run large models on consumer hardware.
- Minimize memory spikes and avoid out-of-memory errors.
- Keep things simple before optimizing for complexity.

---

## 🔭 Roadmap

- [x] GPU/CPU memory tracking
- [x] Multi-level caching (GPU → CPU → Disk)
- [x] Logging to file and console
- [ ] SliceManager with usage-aware eviction
- [ ] Asynchronous slice loading/offloading
- [ ] Automatic slice scheduling
- [ ] Web dashboard / CLI for monitoring slices
- [ ] Pre-built models and benchmarks
---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 👋 Author

Made with ❤️ by [Your Name](https://github.com/yourusername)

```

---

Let me know if you’d like a `.md` file download or if you want to plug in real model examples like `nn.TransformerEncoderLayer` etc.
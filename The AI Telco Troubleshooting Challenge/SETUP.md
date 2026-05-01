# SETUP.md – Environment Documentation

This document describes the hardware, software, and configuration used for data preparation, reasoning trace generation, and model inference in the scope of the **Zindi AI Telco Troubleshooting Challenge**. All work was performed locally to ensure reproducibility, data privacy, and control over compute resources.

## 1. Local Machine for Data Preparation & Inference

### Hardware Specifications

- **Device**: Desktop (DESKTOP-6GO2R35)
- **Processor**: Intel® Core™ Ultra 7 255H (Arrow Lake-H series)
  - 16 cores (6 Performance + 8 Efficient + 2 Low-Power Efficient)
  - Base frequency: 2.00 GHz, Turbo up to ~5.1 GHz
- **Integrated GPU**: Intel® Arc™ Graphics 140T (8 Xe-cores, up to 2.25 GHz)
- **Memory (RAM)**: 32 GB LPDDR5X @ 5600 MT/s (30.9 GB usable)
- **Storage**: 943 GB total (317 GB used at time of writing)
- **Operating System**: Windows 11 (64-bit, latest updates as of January 2026)

### Why this hardware?

The Intel Core Ultra 7 255H with integrated Arc 140T iGPU provides a good balance of CPU multi-threading for data processing and GPU acceleration for LLM inference — critical for generating thousands of reasoning traces efficiently without cloud costs.

## 2. Software Environment

### Python & Dependencies

- **Python version**: 3.10 or 3.11 (recommended)
- **Key libraries used**:
  - `pandas` → data loading, filtering, augmentation
  - `ollama` (Python client) → interaction with local LLM server
  - `tqdm` → progress bars
  - `pathlib`, `argparse` → file handling & CLI
- No additional heavy ML frameworks (PyTorch, Transformers) were required for trace generation.

### LLM Inference Engine

- **Tool**: Ollama (accelerated version via Intel IPEX-LLM)
- **Installation method**: Portable Ollama Zip provided by Intel IPEX-LLM
  - Downloaded from: https://github.com/intel/ipex-llm/releases
  - Folder: `C:\IPEX_OLLAMA` (dedicated to avoid conflict with standard Ollama)
- **Model used**: `qwen2.5:7b-instruct-q4_0`
  - Quantization: Q4_0 (~4.12 GiB)
  - Context length: up to 8192 tokens (set via `num_ctx`)

### GPU Offloading Configuration

All model layers offloaded to iGPU (`OLLAMA_NUM_GPU=999`). Environment variables set before starting server:

```powershell
$env:OLLAMA_NUM_GPU = "999"
$env:ZES_ENABLE_SYSMAN = "1"
$env:SYCL_CACHE_PERSISTENT = "1"
$env:ONEAPI_DEVICE_SELECTOR = "level_zero:0"
.\ollama.exe serve
```

### Performance Achieved

| Configuration | Time for 5 samples | Approx. time per sample | Speedup |
|----------------------------|--------------------|-----------------------------|---------|
| Default Ollama (CPU-only) | ~40 minutes | ~8 minutes | 1× |
| IPEX-LLM + Arc iGPU | ~8 minutes | ~1.6 minutes | ~5× |

GPU utilization during inference: 50–80% on Compute / 3D engine (visible in Windows Task Manager).

## 3. Workflow Summary

1. **Data preparation** → Jupyter notebooks + pandas on CPU cores
2. **Ollama server started** in dedicated IPEX-LLM folder with GPU flags
3. **`generate_reasoning_traces.py` script executed**:
   - Auto-resume from partial CSV
   - Parallel requests possible (via `concurrent.futures`, not yet fully enabled)
   - Added Ollama options: `num_gpu=999`, `num_ctx=8192`, `num_predict=1600`
4. **Output**: Augmented CSV with reasoning traces in `<reasoning>...</reasoning>` tags

## 4. Reproducibility Notes for Hosts / Judges

- All models and tools are open-source / publicly downloadable
- No cloud services or paid APIs used
- Intel Arc iGPU acceleration is hardware-specific but follows official Intel IPEX-LLM documentation
- **Drivers**: Latest Intel Arc & Iris Xe Graphics drivers (≥ 32.x.x recommended)
- The same setup can be replicated on any recent Intel Core Ultra system with Arc graphics

This local environment allowed fast iteration, full control over data, and cost-free scaling of inference — key advantages for high-quality reasoning trace generation in the competition.

---

**Last updated**: January 25, 2026  
**Author**: Kodjo Josué AYITEY (Yehoshua)
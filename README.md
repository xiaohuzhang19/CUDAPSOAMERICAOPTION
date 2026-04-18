# GPU-Accelerated American Option Pricing (CUDA / NVIDIA)

This repository is a **CUDA port** of the OpenCL-based GPU option pricing framework by Leon Xing Li, extended and adapted for NVIDIA GPUs (Google Colab / PyCUDA) by **Xiaohu Zhang**.

Original OpenCL repo: [xiaohuzhang19/Fin_ParallelComputing](https://github.com/xiaohuzhang19/Fin_ParallelComputing)

---

## Papers Overview

This work is based on two research papers:

1. **Using the Graphics Processing Unit to Evaluate American-Style Derivatives**
   - Published in *The Journal of Financial Data Science (JFDS)*
   - Authors: Leon Xing Li, Ren-Raw Chen
   - Explores GPU computing for American option pricing via Monte Carlo simulation (MCS) and Particle Swarm Optimization (PSO) to solve free-boundary PDEs. Achieves significant performance gains over CPU-based methods.

2. **GPU-Accelerated American Option Pricing: The Case of the Longstaff-Schwartz Monte Carlo Model**
   - Published in *The Journal of Derivatives (JOD)*
   - Authors: Leon Xing Li, Ren-Raw Chen, Frank J. Fabozzi
   - Implements the Longstaff-Schwartz Monte Carlo (LSMC) model on GPUs, addressing numerical instability in basis function selection, and optimizes matrix operations for SIMD parallel execution.

---

## What This Fork Adds (Xiaohu Zhang)

The original codebase uses **PyOpenCL**, which runs on Apple Metal, AMD, and NVIDIA GPUs but requires a local OpenCL runtime. This fork ports the entire GPU backend to **PyCUDA**, enabling one-click execution on **Google Colab** with a free NVIDIA T4 GPU.

Key changes:
- All `.cl` OpenCL kernels rewritten as `.cu` CUDA C kernels
- `pyopencl` replaced with `pycuda` throughout (`mc.py`, `pso.py`, `longstaff.py`, `utils.py`)
- Backward-compatible class aliases kept (`PSO_OpenCL_*` → `PSO_CUDA_*`)
- New fused PSO kernel (`knl_source_pso_fusion.cu`) combining searchGrid + fitness + pbest in one kernel launch
- Added `American_option_colab.ipynb` for zero-setup Colab execution
- Empirical study: S&P 500 September 29, 2008 (TARP Rejection Day) — ITM / ATM / vOTM American Put pricing

---

## Features

- **Monte Carlo Simulation on GPU:** Parallel path generation and payoff computation via CUDA
- **Particle Swarm Optimization (PSO) on GPU:** Three GPU variants — hybrid, scalar, and fused — to find the early exercise boundary
- **Longstaff-Schwartz LSMC on GPU:** GPU-accelerated regression (Gauss-Jordan and optimized branchless matrix inversion)
- **NVIDIA / Google Colab ready:** Uses `pycuda.autoinit` for automatic context management — no setup needed on Colab T4
- **Cross-backend price agreement:** CUDA and OpenCL results agree to ≥4 decimal places

---

## Benchmark Results

**Parameters:** S0=100, K=100, r=0.03, σ=0.30, T=1yr | nPath=65,536 · nPeriod=50 · nFish=500

| Method | Price | OpenCL Mac (ms) | Mac Speedup | CUDA NVIDIA (ms) | NVIDIA Speedup |
|---|---:|---:|---:|---:|---:|
| MC — CPU (NumPy)   | 10.3587 | 0.58     | —     | 0.46     | —     |
| MC — GPU           | 10.3587 | 78.36    | 0.0x  | 1194.72  | 0.0x  |
| PSO — CPU (NumPy)  | 10.5814 | 14653.04 | —     | 29010.39 | —     |
| PSO — GPU hybrid   | 10.5814 | 5758.01  | 2.5x  | 6511.96  | 4.5x  |
| PSO — GPU scalar   | 10.5814 | 776.94   | 18.9x | 1047.56  | 27.7x |
| PSO — GPU fusion   | 10.5814 | 759.84   | 19.3x | 1042.02  | 27.8x |
| LSMC — CPU (NumPy) | 10.6055 | 172.62   | —     | 218.26   | —     |
| LSMC — GPU opt     | 10.6054 | 166.34   | 1.0x  | 267.60   | 0.8x  |

> MC-CUDA first-call time includes PyCUDA JIT compilation (~1 sec); actual kernel is <5 ms.

---

## Empirical Study — S&P 500 Sep 29, 2008 (TARP Rejection Day)

**Parameters:** S0=1106.42, r=0.94%, T=30 days | nPath=10,000 · nPeriod=200 · nFish=256

| Case | Market | PSO GPU | PSO Speedup | LSMC GPU | LSMC Speedup |
|---|---:|---:|---:|---:|---:|
| ITM  Put (K=1174.136, σ=0.2824) | 78.9833 | 80.0182 | 8.1x | 79.4481 | 0.4x |
| ATM  Put (K=1106.000, σ=0.2750) | 34.0100 | 34.4599 | 7.7x | 34.3647 | 1.7x |
| vOTM Put (K=987.000,  σ=0.4100) | 10.7400 | 10.8331 | 9.0x | 11.1171 | 1.2x |

> All methods price within ~1.3% of market. See `empirical_study_combined_opencl_cuda.md` for full OpenCL vs CUDA comparison.

---

## How to Run — CUDA (Google Colab, NVIDIA GPU)

**Recommended: use the provided Colab notebook.**

1. Open [src/American_option_colab.ipynb](src/American_option_colab.ipynb) in Google Colab
2. Set runtime: *Runtime → Change runtime type → T4 GPU*
3. Run all cells (Ctrl+F9)

The notebook will:
- Install PyCUDA automatically
- Clone this repo into `/content/CUDAPSOAMERICAOPTION`
- Run benchmarks (Sections 3–6) and the Sep 2008 empirical study (Section 7)

---

## How to Run — OpenCL (Mac / Linux / any GPU)

Use the original OpenCL repo: [xiaohuzhang19/Fin_ParallelComputing](https://github.com/xiaohuzhang19/Fin_ParallelComputing)

### Requirements

```bash
pip install numpy scipy pyopencl matplotlib
```

> On macOS, PyOpenCL uses Apple Metal automatically. On Linux, install an OpenCL ICD for your GPU vendor (NVIDIA, AMD, or Intel).

### Run

```bash
git clone https://github.com/xiaohuzhang19/Fin_ParallelComputing
cd Fin_ParallelComputing/src
python3 American_option.py
```

Or open and run `American_option.ipynb` in Jupyter.

### OpenCL benchmark results (Apple M-series Mac)

| Method | Price | Time (ms) | CPU Speedup |
|---|---:|---:|---:|
| PSO — CPU (NumPy)  | 10.5814 | 14653 | — |
| PSO — GPU scalar   | 10.5814 | 777   | 18.9x |
| PSO — GPU fusion   | 10.5814 | 760   | 19.3x |
| LSMC — CPU (NumPy) | 10.6055 | 173   | — |
| LSMC — GPU opt     | 10.6054 | 166   | 1.0x |

---

## Installation (local CUDA)

To run locally on an NVIDIA machine:

```bash
pip install numpy scipy pycuda matplotlib
```

Requires CUDA toolkit and `nvcc` on PATH.

```bash
git clone https://github.com/xiaohuzhang19/CUDAPSOAMERICAOPTION
cd CUDAPSOAMERICAOPTION/src
python3 -c "from models.utils import checkCUDA; checkCUDA()"
```

---

## Repository Structure

```
src/
├── American_option_colab.ipynb     # Colab notebook (Sections 1–7)
├── American_option.ipynb           # Local notebook
├── models/
│   ├── utils.py                    # CUDAEnv, memory helpers
│   ├── mc.py                       # Monte Carlo (CPU + CUDA GPU)
│   ├── pso.py                      # PSO variants (CPU + CUDA GPU)
│   ├── longstaff.py                # LSMC (CPU + CUDA GPU)
│   └── kernels/
│       ├── mc/                     # MC CUDA kernels (.cu)
│       ├── pso/scalar/             # PSO scalar + fusion kernels (.cu)
│       └── lsmc/                   # LSMC matrix inversion kernels (.cu)
empirical_study_combined_opencl_cuda.md   # OpenCL vs CUDA comparison
```

---

## Citations

**Li, L. X., & Chen, R. R. (2023).** Using the Graphics Processing Unit to Evaluate American-Style Derivatives. *Journal of Financial Data Science.*

**Li, L. X., Chen, R. R., & Fabozzi, F. J. (2024).** GPU-Accelerated American Option Pricing: The Case of the Longstaff-Schwartz Monte Carlo Model. *Journal of Derivatives.*

---

## Contact

Original framework: **Leon Xing Li** — [leonchao@yeah.net](mailto:leonchao@yeah.net)

CUDA port & empirical extension: **Xiaohu Zhang** — [xiaohuzhang19](https://github.com/xiaohuzhang19)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

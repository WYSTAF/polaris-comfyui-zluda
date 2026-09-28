# ComfyUI on an AMD Radeon RX 580 (gfx803) — Windows

This sets up [ComfyUI](https://github.com/Comfy-Org/ComfyUI) so it runs on Polaris
cards (RX 570 / 580 / 590, and the RX 480/470 family) on Windows. It was built by
working through the setup on a real RX 580 8GB, where the usual install path
produces a hardcoded "1 GB VRAM" message and never touches the GPU.

**Please read [Before you start](#before-you-start) first.** This is unofficial
software on top of a dependency that no longer exists upstream. It works, but
nobody is obliged to keep it working.

## What you get

- ComfyUI generating images on an RX 580
- Correct VRAM detection (ComfyUI otherwise hardcodes 1 GiB for DirectML)
- `run_zluda.bat` — one double-click to launch

Real numbers from an RX 580 8GB, SD 1.5, 512×512, 20 steps: **about 100 seconds**
once warm. The first run takes 5–10 minutes longer while ZLUDA compiles its
kernels; those are cached afterwards, so do not delete
`%LOCALAPPDATA%\ZLUDA\ComputeCache`.

## Before you start

You need:

| | |
|---|---|
| OS | Windows 10 22H2 (build 19045) or Windows 11 |
| Python | **3.12, exactly** — the prebuilt wheels do not exist for 3.13 |
| Disk | 12 GB free |
| RAM | 16 GB recommended; 8 GB will struggle |
| Admin rights | Needed once, for the HIP SDK install |

Four things that will otherwise cost you an evening:

1. **Your antivirus will flag ZLUDA.** It uses process-hijack techniques that look
   like malware. This is a known false positive. Add an exclusion for the `zluda`
   folder or Defender will delete it mid-run.
2. **Free up RAM first.** ZLUDA compiles GPU kernels *inside* the Python process
   and needs a couple of GB on top of the model. Starved, it dies with
   `LLVM ERROR: out of memory`. The launcher closes common offenders for you.
3. **The HIP SDK is 1.26 GB and needs admin.** The script extracts only the
   libraries it needs, so nothing is installed system-wide beyond that.
4. **A single `TdrDelay` tweak can help** if you hit GPU timeouts — it needs
   admin, so the script does not do it for you.

## Install

```
git clone <this-repo>
cd rx580-comfyui
python setup.py
```

The script checks its own prerequisites, downloads what it needs, and verifies
the GPU is visible before declaring success. If a step cannot be completed — most
likely the HIP SDK without admin rights — it stops and tells you exactly what to
run by hand.

When it finishes, double-click **`run_zluda.bat`**, then open
<http://127.0.0.1:8188>.

## Which models will work

**SD 1.5 is the practical ceiling.** A single allocation above roughly 1.75 GB
segfaults the process, and that limit comes from the AMD driver, not from
ComfyUI or ZLUDA. Nothing in this repository can raise it.

| Model | Result |
|---|---|
| SD 1.5 (fp16, ~2 GB) | ✅ Works |
| SD 1.5 inpainting, ControlNet | ✅ Works |
| SDXL, Flux, Qwen-Image, anything ≥ 3 GB | ❌ Exceeds the ceiling |

A good first download:
<https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive> →
`v1-5-pruned-emaonly-fp16.safetensors`, into `ComfyUI/models/checkpoints/`.

## How it works

Modern ROCm does not support Polaris, and Microsoft's `torch-directml` — the
official DirectML package — was **deleted from GitHub**, last released September
2024. So the chain is:

```
ComfyUI  →  ZLUDA (translates CUDA calls)  →  HIP SDK  →  patched rocBLAS  →  your card
```

- **ZLUDA** from `lshqqytiger/ZLUDA` — upstream `vosen/ZLUDA` explicitly excludes
  Polaris, so the fork is required.
- **HIP SDK 5.7.1** — supplies `rocblas`/`hipblas` and the compute runtime.
- **Patched rocBLAS** from `advanced-lvl-up/Rx470-...-gfx803-gfx900-fix-AMD-GPU`.
  This is not optional: stock rocBLAS 5.7.1 ships **zero** gfx803 kernels.
- **A `sitecustomize.py` shim** backfilling five torch 2.4 APIs that ComfyUI
  expects but torch 2.2 does not have. torch is pinned to 2.2.x because
  gfx803 cannot use CUDA 12+ or torch ≥ 2.4 under ZLUDA.
- **A patch to ComfyUI itself** for the VRAM fix, the mmap flag, and cuDNN.
  Also upstream as
  [PR #16616](https://github.com/Comfy-Org/ComfyUI/pull/16616).

ComfyUI is pinned to **v0.27.0** for the same reason: v0.37+ needs torch ≥ 2.7 via
`comfy-kitchen`, which gfx803 cannot reach.

## What is not supported

- `bfloat16` — the RX 580 has no hardware support and the emulated path crashes
- bf16 models, including Qwen-Image and SD3
- SageAttention, Triton, xformers, bitsandbytes, cuDNN, `torch.compile`
- Multiple GPUs
- Anything above the ~1.75 GB allocation ceiling

## When it breaks

It will, eventually. ZLUDA is reverse-engineered and unmaintained; torch-directml
is gone from GitHub. If setup fails, check in this order:

1. **Antivirus deleted `zluda.exe`** — restore it and add an exclusion.
2. **`LLVM ERROR: out of memory`** — close everything else and retry.
3. **`Could not find module ... cublas64_11.dll`** — the HIP SDK is missing or the
   `PATH` is wrong; re-run `setup.py`.
4. **`CUDNN_STATUS_INTERNAL_ERROR`** — the launcher should already pass
   `--disable-cudnn`; check you are running the generated `.bat`.
5. **Model runs and then dies** — you are over the VRAM ceiling. Use a smaller
   model.

## Licence and attribution

The ComfyUI patch is contributed upstream under ComfyUI's own licence. ZLUDA,
HIP SDK and rocBLAS are downloaded from their respective authors and remain under
their own terms — this repository distributes none of them.

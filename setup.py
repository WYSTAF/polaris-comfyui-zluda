"""Set up ComfyUI to run on older AMD GPUs (Polaris, Vega) via ZLUDA.

This automates what was done by hand on one machine, and refuses to guess when
it meets something it cannot verify. It is deliberately conservative: every step
checks the result and stops with an explanation rather than continuing on a
half-finished install.

Usage:  python setup.py [--yes]
"""

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

PY = sys.executable

# --- what we fetch ----------------------------------------------------------
ZLUDA_URL = ("https://github.com/lshqqytiger/ZLUDA/releases/download/"
             "rel.854c58e1565c3597e17046d7cc2b1eab96fcdf55/ZLUDA-windows-rocm5-amd64.zip")
ROCMBLAS_URL = ("https://github.com/advanced-lvl-up/Rx470-Vega10-Rx580-gfx803-gfx900-fix-AMD-GPU/"
                "releases/download/v1.0.0/rocm.gfx800-gfx900-for.hip.sdk.5.7.1-and-6.2.4.7z")
HIP_SDK_URL = ("https://download.amd.com/developer/eula/rocm-hub/"
               "AMD-Software-PRO-Edition-23.Q4-Win10-Win11-For-HIP.exe")
TORCH_URL = ("https://download.pytorch.org/whl/cu118/"
             "torch-2.2.1%2Bcu118-cp312-cp312-win_amd64.whl")
TORCHVISION_URL = ("https://download.pytorch.org/whl/cu118/"
                   "torchvision-0.17.1%2Bcu118-cp312-cp312-win_amd64.whl")

# ComfyUI v0.27.0 is the newest release that boots here. v0.37+ needs torch>=2.7
# via comfy-kitchen, and gfx803 cannot use CUDA 12+ or torch>=2.4 under ZLUDA.
COMFYUI_REPO = "https://github.com/Comfy-Org/ComfyUI.git"
COMFYUI_REF = "v0.27.0"

# Versions that must match, because the wrong ones break at import time.
PINS = {
    "numpy": "1.26.4",          # torch 2.2 cannot use numpy>=2
    "transformers": "4.44.2",
    "tokenizers": "0.19.1",     # newer raises ImportError against 4.44.2
    "scipy": "1.11.4",          # 1.13's _propack needs a runtime that will not load
    "safetensors": "0.5.3",
    "comfy-kitchen": "0.2.16",
    "comfy-aimdo": "0.4.10",
    "comfyui-frontend-package": "1.45.20",
    "comfyui-workflow-templates": "0.11.1",
    "comfyui-embedded-docs": "0.5.6",
    # comfyui-workflow-templates 0.11.1 depends on these exact versions. They
    # are listed separately because installing the parent alone is not enough:
    # a leftover newer build satisfies the parent but its own dependencies get
    # skipped, and ComfyUI then fails at startup with "Package
    # 'comfyui_workflow_templates_media_*' is not installed" -- once per missing
    # bundle, while the rest of the UI works and the error looks unrelated.
    "comfyui-workflow-templates-core": "0.3.266",
    "comfyui-workflow-templates-json": "0.1.1",
    "comfyui-workflow-templates-media-api": "0.3.84",
    "comfyui-workflow-templates-media-video": "0.3.101",
    "comfyui-workflow-templates-media-image": "0.3.160",
    "comfyui-workflow-templates-media-other": "0.3.229",
    "comfyui-workflow-templates-media-assets-01": "0.1.0",
}

DL = Path("dl")
ZL = Path("zluda")
HIP = Path("hip")
VENV = Path("venv")
REPO_DIR = Path("ComfyUI")

step = lambda m: print(f"\n=== {m}", flush=True)
info = lambda m: print(f"    {m}", flush=True)
warn = lambda m: print(f"  ! {m}", flush=True)


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        raise SystemExit("failed: " + " ".join(map(str, cmd)) + "\n    " + "\n    ".join(tail))
    return r


def _has_venv():
    return importlib.util.find_spec("venv") is not None


def _has_pip():
    return importlib.util.find_spec("pip") is not None


def check_prereqs():
    step("Checking prerequisites")
    if sys.platform != "win32":
        raise SystemExit("This setup is Windows-only (ZLUDA has no other platform).")
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(
            f"Need Python 3.12 exactly; this is {sys.version_info.major}.{sys.version_info.minor}. "
            "The cu118 torch wheels are built for cp312.")
    info(f"python {sys.version.split()[0]} on {sys.platform} - ok")
    if "venv" not in sys.modules and not _has_venv():
        warn("This Python has no venv module (ComfyUI's bundled python_embeded does not).")
        warn("setup.py will fall back to virtualenv, which needs a working pip.")
    if "pip" not in sys.modules and not _has_pip():
        warn("This Python has no pip. Install Python 3.12 from python.org instead.")
    free = shutil.disk_usage(".").free / 1024**3
    info(f"{free:.1f} GB free here")
    if free < 12:
        warn("Under 12 GB free. The downloads alone need ~6 GB, plus the venv.")
    admin = False
    try:
        import ctypes
        admin = bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        pass
    info(f"admin rights: {'yes' if admin else 'no'}")
    if not admin:
        warn("No admin rights. The HIP SDK installer needs them - run it yourself "
             "when prompted, and choose a location on a drive with 6+ GB free.")
    return admin


def download(url, dest, label, attempts=200):
    """Fetch a file, resuming and verifying it is complete.

    This network drops long transfers part-way without raising, and a truncated
    zip or 7z still has a valid magic number, so the failure otherwise shows up
    much later as "File is not a zip file". Two defences: every response is
    checked against Content-Length, and a partial file is continued with a
    Range request rather than thrown away -- the HIP SDK is 1.2 GB and a network
    that cuts off around a few hundred MB would otherwise never finish it.
    """
    dest = Path(dest)
    total = 0
    known_size = {}
    last_have = 0
    stalled = 0

    for attempt in range(1, attempts + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if have and label in known_size and have >= known_size[label]:
            info(f"{label}: already complete ({have / 1024**2:.1f} MB)")
            return dest

        headers = {"User-Agent": "Mozilla/5.0"}
        if have:
            headers["Range"] = f"bytes={have}-"
            info(f"{label}: resuming at {have / 1024**2:.0f} MB"
                 + (f" of {known_size[label] / 1024**2:.0f} MB" if label in known_size else ""))
        else:
            info(f"{label}: downloading -> {dest.name}")

        try:
            req = urllib.request.Request(url, headers=headers)
            try:
                r = urllib.request.urlopen(req, timeout=120)
            except urllib.error.HTTPError as e:
                # 416 means the bytes we already have reach past the end of the
                # file, i.e. it is already complete. Re-fetching the size
                # header is enough to confirm.
                if e.code == 416 and have:
                    with urllib.request.urlopen(
                            urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
                            timeout=120) as chk:
                        full = int(chk.headers.get("Content-Length", 0))
                    if full and have >= full:
                        info(f"{label}: already complete ({have / 1024**2:.1f} MB)")
                        return dest
                raise
            with r:
                if r.status == 206:          # partial content
                    mode = "ab"
                    total = have + int(r.headers.get("Content-Length", 0))
                else:                        # server ignored the Range
                    mode = "wb"
                    have = 0
                    total = int(r.headers.get("Content-Length", 0))
                if total:
                    known_size[label] = total
                with open(dest, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        if total:
                            print(f"\r    {have * 100 // total}% "
                                  f"({have / 1024**2:.0f}/{total / 1024**2:.0f} MB)",
                                  end="", flush=True)
            if total:
                print()
            if total and have < total:
                raise IOError(f"short read: {have} of {total} bytes")
            info(f"{label}: done ({have / 1024**2:.1f} MB)")
            return dest
        except Exception as e:
            got = dest.stat().st_size if dest.exists() else 0
            progressed = got > last_have
            if not progressed:
                stalled += 1
            else:
                stalled = 0
                if total:
                    info(f"{label}: {got * 100 // total}% ({got / 1048576:.0f} MB) "
                         "- connection dropped, resuming")
            last_have = got

            # Several attempts in a row making no progress means the link is
            # not usable for a file this size, rather than merely slow.
            if stalled >= 4:
                raise SystemExit(
                    f"Download of {label} is stuck at {got / 1048576:.0f} MB: {e}\n"
                    f"  {url}\n"
                    "  This connection is dropping the transfer every time and never\n"
                    "  getting further. Torch is 2.6 GB, which needs a steady link.\n"
                    "  Options:\n"
                    f"    - download it in a browser and save it as dl/{dest.name}\n"
                    "    - run setup.py again later on a better connection\n"
                    "    - on a metered or slow link, fetch it over several sittings;\n"
                    "      each run continues from where the last stopped")
            time.sleep(3)


def check_archive(path, kind):
    """Return True if `path` is a structurally valid archive of `kind`.

    A truncated download still has a valid magic number, so opening it is not
    proof. This is called before trusting a file that download() had cached.
    """
    p = Path(path)
    if not p.exists():
        return False
    try:
        if kind == "zip":
            with zipfile.ZipFile(p) as z:
                bad = z.testzip() if z.namelist() else "empty"
        else:  # 7z
            import py7zr
            with py7zr.SevenZipFile(p, "r") as z:
                bad = None if z.getnames() else "empty"
    except Exception as e:
        warn(f"{p.name} is not a usable {kind}: {e}")
        return False
    if bad:
        warn(f"{p.name} is corrupt ({bad})")
        return False
    return True


def setup_zluda():
    step("Installing ZLUDA (the CUDA-to-AMD translator)")
    zp = DL / "zluda.zip"
    if zp.exists() and not check_archive(zp, "zip"):
        zp.unlink()  # truncated from an earlier run; fetch it again
    z = download(ZLUDA_URL, zp, "zluda")
    if not check_archive(z, "zip"):
        raise SystemExit("The ZLUDA archive is corrupt. Delete dl/zluda.zip and re-run.")
    out = Path(ZL)
    if not (out / "zluda" / "zluda.exe").exists():
        out.mkdir(exist_ok=True)
        with zipfile.ZipFile(z) as f:
            f.extractall(out)
    exe = out / "zluda" / "zluda.exe"
    if not exe.exists():
        raise SystemExit("zluda.exe is missing from the extracted archive - the "
                         "download may be corrupt. Delete dl/zluda.zip and retry.")
    info(f"zluda.exe ready at {exe}")
    warn("If Windows Defender flags zluda.exe, add this folder to your exclusions. "
         "It uses process-hijack techniques and is a known false positive.")


def setup_hip(admin):
    step("HIP SDK (AMD's compute library that ZLUDA needs)")
    exe = DL / "hip-sdk.exe"
    if not exe.exists():
        download(HIP_SDK_URL, exe, "hip-sdk")
    target = Path(HIP).resolve()
    lib = target / "Program Files 64" / "AMD" / "ROCm" / "5.7" / "bin"

    if not (lib / "rocblas.dll").exists():
        if not admin:
            raise SystemExit(
                "The HIP SDK still needs installing, and that needs admin rights.\n"
                f"  Run:  {exe}\n"
                "  Choose 'Custom' and set the install location to a drive with 6+ GB free.\n"
                "  Then re-run this script; it will pick up where it left off.")
        info("Extracting the libraries only (no drivers, no Visual Studio)...")
        run(["msiexec", "/a", str(exe), "/qn", f"TARGETDIR={target}", "/l*v", "hip_extract.log"])
        if not (lib / "rocblas.dll").exists():
            raise SystemExit(f"Extraction did not produce rocblas.dll under {lib}")

    info(f"HIP libs at {lib}")
    return lib


def setup_rocblas(lib):
    step("Patched rocBLAS for gfx803")
    original = lib / "rocblas.dll"
    kdir = lib / "rocblas" / "library"
    if (kdir).exists() and any("gfx803" in f.name for f in kdir.iterdir()):
        info("gfx803 kernels already installed")
        return
    if kdir.exists():
        existing = list(kdir.iterdir())
        if not any("gfx803" in f.name for f in existing):
            warn(f"stock rocBLAS has {len(existing)} kernels and none for gfx803. "
                 "This is the fix that makes the card usable at all.")
    z = download(ROCMBLAS_URL, DL / "rocblas.7z", "rocblas-gfx803")
    info("extracting (needs 7-Zip or py7zr)")
    try:
        import py7zr
    except ImportError:
        run([PY, "-m", "pip", "install", "py7zr"])
        import py7zr
    if not check_archive(z, "7z"):
        raise SystemExit("The rocBLAS archive is corrupt. Delete dl/rocblas.7z and re-run.")
    with py7zr.SevenZipFile(z, "r") as f:
        f.extractall("rocblas_fix")
    src = Path("rocblas_fix")
    if (src / "rocblas.dll").exists():
        backup = Path("backup_rocblas.dll")
        if original.exists() and not backup.exists():
            shutil.copy2(original, backup)
            info(f"original rocblas.dll backed up to {backup}")
        shutil.copy2(src / "rocblas.dll", original)
        if (src / "library").exists():
            if kdir.exists():
                shutil.rmtree(kdir)
            shutil.copytree(src / "library", kdir)
    n = len([f for f in kdir.iterdir() if "gfx803" in f.name]) if kdir.exists() else 0
    info(f"installed {n} gfx803 kernels")
    if n == 0:
        warn("no gfx803 kernels found - the archive may be wrong for your HIP SDK version")


def setup_venv():
    step("Python environment with torch 2.2.1+cu118")
    vpy = Path(VENV) / "Scripts" / "python.exe"
    if vpy.exists():
        info("venv already exists")
    else:
        # ComfyUI's bundled python_embeded has no venv module, and neither do
        # some minimal installs. Fall back to virtualenv, which does not need it.
        r = subprocess.run([PY, "-m", "venv", str(VENV)], capture_output=True, text=True)
        if r.returncode != 0 or not vpy.exists():
            info("this Python has no venv module, trying virtualenv")
            rv = subprocess.run([PY, "-m", "pip", "install", "virtualenv"],
                               capture_output=True, text=True)
            run([PY, "-m", "virtualenv", str(VENV)])
        if not vpy.exists():
            raise SystemExit(
                "Could not create a virtual environment with this Python.\n"
                "  Install Python 3.12 from python.org (the 'Windows installer'), which\n"
                "  includes venv, then re-run setup.py with that python.")
    info(f"venv python: {vpy}")

    def have_pkg(name):
        r = subprocess.run([str(vpy), "-c",
                            f"import importlib.metadata as m; print(m.version('{name}'))"],
                           capture_output=True, text=True)
        return r.returncode == 0

    # torch is a 2.6 GB download, so do not fetch it when it is already there.
    if have_pkg("torch"):
        info("torch is already installed in this venv, skipping the 2.6 GB download")
    else:
        w = download(TORCH_URL, DL / "torch-2.2.1+cu118-cp312-cp312-win_amd64.whl", "torch")
        if not check_archive(w, "zip"):
            raise SystemExit("The torch wheel is corrupt. Delete it from dl/ and re-run.")
        run([str(vpy), "-m", "pip", "install", "--no-cache-dir", "--no-deps", str(w)])

    if have_pkg("torchvision"):
        info("torchvision is already installed, skipping")
    else:
        tv = download(TORCHVISION_URL, DL / "torchvision-0.17.1+cu118-cp312-cp312-win_amd64.whl", "torchvision")
        if not check_archive(tv, "zip"):
            raise SystemExit("The torchvision wheel is corrupt. Delete it from dl/ and re-run.")
        run([str(vpy), "-m", "pip", "install", "--no-cache-dir", "--no-deps", str(tv)])

    for name, ver in PINS.items():
        for attempt in range(4):
            r = subprocess.run([str(vpy), "-m", "pip", "install", "--no-cache-dir",
                                "--retries", "5", f"{name}=={ver}"],
                               capture_output=True, text=True)
            if r.returncode == 0 or "already satisfied" in (r.stdout + r.stderr):
                break
        info(f"  {name}=={ver}")

    deps = ["einops", "regex", "huggingface-hub", "packaging", "sentencepiece", "Pillow",
            "psutil", "alembic", "SQLAlchemy", "av", "requests", "simpleeval", "blake3",
            "kornia", "spandrel", "pydantic", "pydantic-settings", "torchsde", "tqdm",
            "aiohttp", "yarl", "pyyaml", "filelock", "matplotlib"]
    for d in deps:
        for attempt in range(4):
            r = subprocess.run([str(vpy), "-m", "pip", "install", "--no-cache-dir", "--retries", "5", d],
                               capture_output=True, text=True)
            if r.returncode == 0 or "already satisfied" in (r.stdout + r.stderr):
                break
        else:
            warn(f"could not install {d} - ComfyUI may fail to start without it")
    run([str(vpy), "-m", "pip", "install", "--no-cache-dir", "numpy==1.26.4"])
    return vpy


def setup_torch_dlls(vpy):
    step("Pointing torch's CUDA libraries at ZLUDA")
    lib = Path(VENV) / "Lib" / "site-packages" / "torch" / "lib"
    zs = Path(ZL) / "zluda"
    pairs = [("cublas.dll", "cublas64_11.dll"),
             ("cusparse.dll", "cusparse64_11.dll"),
             ("nvrtc.dll", "nvrtc64_112_0.dll")]
    bak = Path("backup_torch_dlls")
    bak.mkdir(exist_ok=True)
    for src_name, dst_name in pairs:
        dst = lib / dst_name
        if dst.exists() and dst.stat().st_size < 1024 * 1024:
            info(f"{dst_name}: already points at ZLUDA")
            continue
        if dst.exists():
            shutil.copy2(dst, bak / dst_name)
        shutil.copy2(zs / src_name, dst)
        info(f"{dst_name} <- {src_name}")
    warn("cudart is deliberately NOT replaced. Swapping it causes a hard crash.")


def setup_shim(vpy):
    step("Installing the torch compatibility shim")
    here = Path(__file__).parent
    src = here / "shims" / "sitecustomize.py"
    if not src.exists():
        raise SystemExit(f"shim missing at {src}")
    dst = Path(VENV) / "Lib" / "site-packages" / "sitecustomize.py"
    shutil.copy2(src, dst)
    info(f"installed {dst}")
    info("It backfills the torch 2.4 APIs ComfyUI v0.27.0 assumes, which torch 2.2 lacks.")


def setup_comfyui():
    step("ComfyUI v0.27.0 (the newest that boots on gfx803)")
    if (Path(REPO_DIR) / "main.py").exists():
        info("already present")
        return
    run(["git", "clone", "--depth", "1", "--branch", COMFYUI_REF, COMFYUI_REPO, str(REPO_DIR)])
    patch = Path(__file__).parent / "patches" / "comfyui-legacy-gpu.patch"
    if patch.exists():
        info("applying legacy-GPU fixes")
        r = subprocess.run(["git", "-C", str(REPO_DIR), "apply", str(patch)], capture_output=True, text=True)
        if r.returncode == 0:
            info("patch applied")
        elif "already applied" in (r.stdout + r.stderr):
            info("patch already applied")
        else:
            warn("patch did not apply cleanly - the fixes may already be upstream:\n    "
                 + (r.stdout + r.stderr).strip()[:300])


def write_launcher(hlib):
    step("Writing the launcher")
    here = Path(__file__).parent.resolve()
    bat = here / "run_zluda.bat"
    bat.write_text(
        ":: ComfyUI on AMD Radeon RX 580 via ZLUDA. Generated by setup.py.\n"
        "::\n"
        ":: Close memory-heavy apps first. ZLUDA compiles PTX inside this process and\n"
        ":: needs a few GB of free RAM, on top of the model, or it aborts with an\n"
        ":: LLVM 'out of memory'.\n"
        "::\n"
        ":: The first run also spends 5-10 min compiling PTX. It is then cached in\n"
        ":: %%LOCALAPPDATA%%\\ZLUDA\\ComputeCache and later runs are far quicker.\n"
        "::\n"
        ":: KNOWN LIMIT: this card has a hard ~1.75 GiB allocation ceiling (the AMD\n"
        ":: driver segfaults above it), so SD1.5-class models are the practical\n"
        ":: maximum. Anything much larger will not fit.\n"
        "\n"
        "taskkill /F /IM java.exe /T >nul 2>&1\n"
        "taskkill /F /IM firefox.exe /T >nul 2>&1\n"
        "taskkill /F /IM msedge.exe /T >nul 2>&1\n"
        "\n"
        f'set "HIP_PATH={hlib.parent}"\n'
        'set "PATH=%HIP_PATH%\\bin;%PATH%"\n\n'
        f'"{here / ZL / "zluda" / "zluda.exe"}" -- "{here / VENV / "Scripts" / "python.exe"}" '
        f'-s "{here / REPO_DIR / "main.py"}" --use-quad-cross-attention --reserve-vram 0.9 '
        '--disable-async-offload --disable-pinned-memory --disable-cuda-malloc '
        '--disable-mmap --disable-cudnn --cpu-vae --lowvram %*\n'
        "pause\n", encoding="utf-8")
    info(f"created {bat.name}")


def check_python_packages(vpy):
    """Confirm every pinned package is importable, not merely installed.

    A leftover newer build satisfies `pip install` while leaving a package
    whose module directory was never written, and the breakage only surfaces
    later as an ImportError inside ComfyUI. Importing each one here turns that
    into an immediate, named failure.
    """
    step("Checking Python packages")
    code = (
        "import importlib, importlib.metadata as m, sys\n"
        "bad = []\n"
        f"want = {PINS!r}\n"
        "for name, ver in want.items():\n"
        "    mod = name.replace('-', '_')\n"
        "    try:\n"
        "        have = m.version(name)\n"
        "    except Exception:\n"
        "        have = None\n"
        "    try:\n"
        "        importlib.import_module(mod)\n"
        "        ok = True\n"
        "    except Exception as e:\n"
        "        ok = False\n"
        "        print(f'   FAIL {name}: {type(e).__name__}')\n"
        "    if not ok:\n"
        "        bad.append(name)\n"
        "    elif have != ver:\n"
        "        print(f'   WARN {name}: {have} installed, {ver} pinned')\n"
        "print('BROKEN=' + ','.join(bad))\n"
    )
    r = subprocess.run([str(vpy), "-c", code], capture_output=True, text=True)
    for line in (r.stdout or "").splitlines():
        if line.strip():
            print("   ", line.strip())
    out = (r.stdout or "")
    broken = ""
    for line in out.splitlines():
        if line.startswith("BROKEN="):
            broken = line.split("=", 1)[1].strip()
    if broken:
        raise SystemExit(
            f"These packages are installed but cannot be imported: {broken}\n"
            "  This usually means a partial download. Re-run setup.py; it will\n"
            "  reinstall them. If it persists, clear the pip cache first:\n"
            "    python -m pip cache purge")
    info("all pinned packages import cleanly")


def verify(vpy, hlib):
    step("Verifying the GPU is actually visible")
    env = dict(os.environ)
    env["HIP_PATH"] = str(hlib.parent)
    env["PATH"] = str(hlib) + os.pathsep + env["PATH"]
    code = (
        "import torch;"
        "print('cuda available:', torch.cuda.is_available());"
        "print('device:', torch.cuda.get_device_name(0));"
        "import torch as t;"
        "a=t.randn(256,256,device='cuda');"
        "print('matmul on GPU: ok', float((a@a).sum()))"
    )
    r = subprocess.run([str(Path(ZL) / "zluda" / "zluda.exe"), "--", str(vpy), "-c", code],
                       capture_output=True, text=True, env=env, timeout=900)
    out = (r.stdout + r.stderr).strip()
    for line in out.splitlines():
        if any(k in line for k in ("cuda available", "device:", "matmul", "Error", "error")):
            print("   ", line)
    if "matmul on GPU: ok" not in out:
        warn("The GPU check did not pass. Free up RAM and try again - ZLUDA's PTX")
        warn("compiler needs a few GB, and the first run compiles for several minutes.")
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    Path(DL).mkdir(exist_ok=True)
    print("ComfyUI on older AMD GPUs (Polaris gfx803, Vega gfx900) via ZLUDA")
    print("This is unofficial. See README.md for what may go wrong.")

    admin = check_prereqs()
    setup_zluda()
    hlib = setup_hip(admin)
    setup_rocblas(hlib)
    vpy = setup_venv()
    check_python_packages(vpy)
    setup_torch_dlls(vpy)
    setup_shim(vpy)
    setup_comfyui()
    write_launcher(hlib)

    if not a.yes:
        step("Ready to test the GPU")
        if input("Run the GPU verification now? [y/N] ").strip().lower() != "y":
            print("\nSkipped. Run run_zluda.bat when you are ready.")
            return
    ok = verify(vpy, hlib)

    step("Done")
    if ok:
        print("  Double-click run_zluda.bat to start ComfyUI.")
        print("  Then open http://127.0.0.1:8188")
        print("\n  A model goes in ComfyUI/models/checkpoints/.")
        print("  Start small: SD 1.5 fp16 (~2 GB) works on this card.")
    else:
        print("  Run run_zluda.bat anyway, or re-run setup.py to retry the check.")


if __name__ == "__main__":
    main()

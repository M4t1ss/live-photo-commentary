"""
Windows-only: waits for a just-(re)installed `torch` package to become
reliably importable.

torch may have been installed or reinstalled moments before this process
started (CUDA venv setup via `install_cuda_torch` + `restart_backend`).
NTFS's directory-entry index can lag the on-disk contents it indexes for a
short window after such a reinstall, which surfaces as three distinct
import failures depending on exactly which file's directory entry hasn't
caught up yet:

 1. `torch/__init__.py`'s entry is stale: `find_spec("torch")` resolves the
    package directory but doesn't see `__init__.py`, returning a namespace
    package (`spec.origin is None`) instead of raising.
 2. The compiled `torch/_C` extension module's entry is stale: torch's own
    `__init__.py` falls back to the `torch/_C/` stub-only directory (shipped
    for type checkers) and raises "Failed to load PyTorch C extensions: ...".
 3. `torch\\lib\\torch.dll` (or a dependency)'s entry is stale: the DLL
    loader can't find it even though `__init__.py` is visible, raising
    `OSError` with `winerror == 126`.

`importlib.invalidate_caches()` flushes Python's `FileFinder` cache, forcing
a fresh directory read on the next attempt; each retry also clears any
partially-imported `torch` modules from `sys.modules` so the next attempt
starts clean.

Call `wait_for_torch()` before the first `import torch` in any entry point —
it's a no-op off Windows.
"""
import sys


def wait_for_torch() -> None:
    if sys.platform != "win32":
        return

    import time
    import importlib
    import importlib.util

    for i in range(120):
        importlib.invalidate_caches()
        try:
            spec = importlib.util.find_spec("torch")
            if spec is not None and spec.origin is not None:
                try:
                    import torch  # noqa: F401  (also exercises DLL loading)
                    return
                except OSError as e:
                    if getattr(e, "winerror", None) != 126:
                        raise  # unexpected error — don't suppress
                    _clear_torch_modules()
                except ImportError as e:
                    if "Failed to load PyTorch C extensions" not in str(e):
                        raise  # unexpected error — don't suppress
                    _clear_torch_modules()
        except Exception:
            pass

        if i == 0:
            print("[lpc] waiting for torch to become visible…",
                  flush=True, file=sys.stderr)
        time.sleep(0.5)

    print("[lpc] gave up waiting for torch; import may fail",
          flush=True, file=sys.stderr)


def _clear_torch_modules() -> None:
    # torch.dll / torch\_C not yet visible; remove any partial state so the
    # next attempt starts clean.
    for mod in [k for k in sys.modules if k == "torch" or k.startswith("torch.")]:
        sys.modules.pop(mod, None)

#!/usr/bin/env python3
"""
Wire a pre-built pyopenjtalk wheel into the bundled pyproject.toml so that
uv sync on the end-user's machine resolves it from vendor/ instead of PyPI
(which has no pre-built wheels for any platform).

Usage: python3 tools/patch_pyopenjtalk.py <backend-dir>
"""
import pathlib, re, sys


def main():
    backend_dir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("src-tauri/resources/backend")
    vendor_dir = backend_dir / "vendor"

    wheels = sorted(vendor_dir.glob("pyopenjtalk-*.whl"))
    if not wheels:
        sys.exit(f"no pyopenjtalk wheel found in {vendor_dir}")
    whl = wheels[0].name
    print("wiring", whl)

    pj = backend_dir / "pyproject.toml"
    text = pj.read_text()

    # [tool.uv.sources] only applies to direct dependencies, so make pyopenjtalk
    # explicit in the dependencies list if it is not already there.
    if '"pyopenjtalk"' not in text:
        text = re.sub(r'("misaki\[en,ja\][^"]*",)', r'\1\n  "pyopenjtalk",', text)
    text = re.sub(r"\n\[tool\.uv\.sources\].*", "", text, flags=re.DOTALL)
    text += f'\n[tool.uv.sources]\npyopenjtalk = {{path = "vendor/{whl}"}}\n'
    pj.write_text(text)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Install V3: split before QKV, then defer full-batch attention preparation. Default: dry run."""
import argparse
import difflib
import hashlib
import importlib.util
import os
from pathlib import Path
import stat
import tempfile


KNOWN = {
    "9f2a521f6eeca5b54f47de54129c6ce2c4ddf5d450e8ab10f111b86d0ba939d8": "audited original",
    "da1982549c71ff6f2fa975e12117161ebbfdb2b3ffcff007e53dd7446d3d4a8e": "previous dual-only patch",
    "087c5a3d4f3087c3d82cdd30392be0f51278307896db1bdca2ee62d53e604ff9": "previous defer-only patch",
    "15d723d1cbd070c0ad3a6536fd78640cc0962b00e5bedd68e6737acad076d31f": "combined V2 (full-batch front preparation)",
}
EXPECTED_NEW = "82c57f3bc32c7c13bd7af53ce1c574f5922f28a4b476f522a3a16ff1ee4fac98"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def find_installed_source():
    spec = importlib.util.find_spec("vllm_ascend")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("Cannot locate vllm_ascend. Use the service Python or --source /actual/path/sfa_v1.py")
    return Path(next(iter(spec.submodule_search_locations))) / "attention" / "sfa_v1.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Installed file to update")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--show-diff", action="store_true")
    args = parser.parse_args()
    replacement_path = Path(__file__).resolve().with_name("sfa_v1.py")
    replacement = replacement_path.read_bytes()
    if digest(replacement) != EXPECTED_NEW:
        raise SystemExit("Packaged sfa_v1.py checksum mismatch; refusing to install")
    compile(replacement, str(replacement_path), "exec")
    target = (args.source or find_installed_source()).resolve()
    original = target.read_bytes()
    current = digest(original)
    print(f"Installed file: {target}\nSHA256: {current}")
    if current == EXPECTED_NEW:
        print("Combined patch already installed; unchanged.")
        return
    if current not in KNOWN:
        raise SystemExit("Unsupported installed source; unchanged. Adapt this exact source before installing.")
    print(f"Recognized: {KNOWN[current]}")
    if args.show_diff:
        print("".join(difflib.unified_diff(original.decode().splitlines(True), replacement.decode().splitlines(True),
                                        fromfile=str(target), tofile=str(replacement_path))))
    if not args.apply:
        print("Dry run passed; no files changed. Stop all workers, then rerun with --apply.")
        return
    backup = target.with_name(target.name + ".before_dual_defer_v3")
    with backup.open("xb") as f:
        f.write(original)
    mode = stat.S_IMODE(target.stat().st_mode)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".sfa-dual-", delete=False) as f:
            tmp_path = Path(f.name)
            f.write(replacement)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.chmod(mode)
        os.replace(tmp_path, target)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    print(f"Installed combined patch. Backup: {backup}")
    print("Restart every worker with VLLM_ASCEND_SFA_DUAL_BATCH=1 and enable_mlapo=false.")
    print("Use FULL_DECODE_ONLY and recapture graphs. NPU validation is still required.")


if __name__ == "__main__":
    main()

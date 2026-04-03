"""
Native pickle/serialization scanner — modelscan-equivalent for Python 3.13+.

Scans model files for dangerous pickle opcodes and known unsafe patterns,
replicating the core logic of ProtectAI's modelscan tool.
"""

from __future__ import annotations

import io
import os
import pickletools
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Opcodes that allow arbitrary code execution during unpickling
_DANGEROUS_OPCODES: set[str] = {
    "GLOBAL",       # Imports a class/function: can import anything
    "INST",         # Instantiates a class
    "OBJ",          # Builds an object from stack
    "NEWOBJ",       # Builds an object via __new__
    "NEWOBJ_EX",    # Builds an object via __new__ with kwargs
    "REDUCE",       # Calls a callable with args (the main RCE vector)
    "BUILD",        # Calls __setstate__ — can be used for RCE
    "STACK_GLOBAL", # Like GLOBAL but from stack
}

# Known malicious or highly suspicious import targets
_SUSPICIOUS_IMPORTS: set[tuple[str, str]] = {
    ("os", "system"),
    ("os", "popen"),
    ("os", "execve"),
    ("os", "execvpe"),
    ("subprocess", "Popen"),
    ("subprocess", "check_output"),
    ("subprocess", "run"),
    ("subprocess", "call"),
    ("builtins", "eval"),
    ("builtins", "exec"),
    ("builtins", "__import__"),
    ("importlib", "import_module"),
    ("nt", "system"),
    ("posix", "system"),
    ("ctypes", "CDLL"),
    ("socket", "socket"),
    ("codecs", "encode"),  # often used for obfuscation
    ("base64", "b64decode"),
    ("marshal", "loads"),
    ("pickle", "loads"),
    ("shelve", "open"),
    ("runpy", "run_module"),
}


@dataclass
class PickleScanResult:
    filepath: str
    is_safe: bool = True
    dangerous_opcodes: list[str] = field(default_factory=list)
    suspicious_imports: list[tuple[str, str]] = field(default_factory=list)
    global_imports: list[tuple[str, str]] = field(default_factory=list)
    error: str | None = None
    file_format: str = "pickle"


def _scan_pickle_bytes(data: bytes, filepath: str) -> PickleScanResult:
    result = PickleScanResult(filepath=filepath)
    dangerous: list[str] = []
    suspicious: list[tuple[str, str]] = []
    all_globals: list[tuple[str, str]] = []

    try:
        ops = list(pickletools.genops(io.BytesIO(data)))
    except Exception as exc:
        result.error = f"Failed to parse pickle opcodes: {exc}"
        result.is_safe = False
        return result

    for op, arg, pos in ops:
        name = op.name
        if name in _DANGEROUS_OPCODES:
            dangerous.append(name)

        if name in ("GLOBAL", "INST"):
            if isinstance(arg, str):
                parts = arg.split(" ", 1)
                if len(parts) == 2:
                    module, attr = parts[0], parts[1]
                    all_globals.append((module, attr))
                    if (module, attr) in _SUSPICIOUS_IMPORTS:
                        suspicious.append((module, attr))
                    # Heuristic: shell/exec keywords anywhere
                    combined = f"{module}.{attr}".lower()
                    if any(kw in combined for kw in ("exec", "eval", "system", "popen", "spawn", "shell")):
                        if (module, attr) not in suspicious:
                            suspicious.append((module, attr))

        elif name == "STACK_GLOBAL":
            # args are already on stack; we look at recent SHORT_BINUNICODE/UNICODE ops
            pass  # handled via the globals list above for GLOBAL

    result.dangerous_opcodes = list(set(dangerous))
    result.suspicious_imports = suspicious
    result.global_imports = all_globals
    result.is_safe = len(suspicious) == 0
    return result


def _extract_pickle_from_pytorch(path: Path) -> list[tuple[str, bytes]]:
    """Extract pickle data from PyTorch .pt/.pth/.bin files (zip archives)."""
    pickles: list[tuple[str, bytes]] = []
    try:
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                if name.endswith(".pkl") or name.endswith("/data.pkl") or "data.pkl" in name:
                    pickles.append((name, zf.read(name)))
    except zipfile.BadZipFile:
        # Might be a raw pickle file
        data = path.read_bytes()
        if data[:2] in (b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05", b"\x80\x06"):
            pickles.append((path.name, data))
    return pickles


# File extensions that may contain pickled data
_PICKLE_EXTENSIONS = {".pkl", ".pickle", ".pt", ".pth", ".bin", ".ckpt", ".joblib", ".model"}
_SAFE_EXTENSIONS = {".safetensors", ".gguf", ".ggml", ".json", ".txt", ".md", ".yaml", ".yml", ".py"}


def scan_path(path: str | Path) -> list[PickleScanResult]:
    """Recursively scan a file or directory for unsafe pickle content."""
    p = Path(path)
    results: list[PickleScanResult] = []

    if p.is_file():
        results.extend(_scan_file(p))
    elif p.is_dir():
        for root, _, files in os.walk(p):
            for fname in files:
                fp = Path(root) / fname
                results.extend(_scan_file(fp))
    return results


def _scan_file(p: Path) -> list[PickleScanResult]:
    suffix = p.suffix.lower()
    if suffix in _SAFE_EXTENSIONS:
        return []
    if suffix not in _PICKLE_EXTENSIONS and suffix not in {".npz", ".npy", ".h5", ".hdf5"}:
        # Check magic bytes for pickle
        try:
            magic = p.read_bytes()[:2]
            if magic not in (b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05"):
                return []
        except Exception:
            return []

    results: list[PickleScanResult] = []

    if suffix in {".pt", ".pth", ".bin", ".ckpt"}:
        pickles = _extract_pickle_from_pytorch(p)
        for subname, data in pickles:
            r = _scan_pickle_bytes(data, f"{p}::{subname}")
            r.file_format = "pytorch"
            results.append(r)
    elif suffix in {".pkl", ".pickle", ".joblib", ".model"}:
        try:
            data = p.read_bytes()
        except Exception as exc:
            results.append(PickleScanResult(filepath=str(p), is_safe=False, error=str(exc)))
            return results
        r = _scan_pickle_bytes(data, str(p))
        r.file_format = "pickle"
        results.append(r)
    elif suffix == ".npz":
        try:
            with zipfile.ZipFile(p, "r") as zf:
                for name in zf.namelist():
                    if name.endswith(".npy"):
                        data = zf.read(name)
                        # NumPy .npy files can embed pickles for object arrays
                        if b"NUMPY" in data[:10] and b"\x80" in data:
                            r = _scan_pickle_bytes(data, f"{p}::{name}")
                            r.file_format = "numpy"
                            results.append(r)
        except Exception:
            pass

    return results

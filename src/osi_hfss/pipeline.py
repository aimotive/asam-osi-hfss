import functools
import hashlib
import json
import subprocess
from pathlib import Path

from osi_hfss import modules

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "build" / "data"
PIPELINE = DATA_DIR / "pipeline.json"


def steps() -> dict[str, dict]:
    return json.loads(PIPELINE.read_text())["files"]


def is_input(name: str) -> bool:
    return (DATA_DIR / name).is_file() and name not in steps()


def names() -> list[str]:
    inputs = [p.name for p in DATA_DIR.iterdir() if p.is_file() and p.name != PIPELINE.name]
    return sorted(set(inputs) | set(steps()))


@functools.cache
def _file_hash(path: Path, mtime_ns: int) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash(path: Path) -> str:
    return _file_hash(path, path.stat().st_mtime_ns)


def _module_hash(module: str) -> str:
    launcher = modules.executable(module)
    files = [launcher, *sorted((launcher.parent / "lib").glob("*"))]
    return hashlib.sha256("".join(_hash(f) for f in files if f.is_file()).encode()).hexdigest()


def path(name: str) -> Path:
    step = steps().get(name)
    if step is None:
        if not (DATA_DIR / name).is_file():
            raise FileNotFoundError(f"{name} is neither an input in data/ nor described in {PIPELINE.name}")
        return DATA_DIR / name
    inputs = {option: path(file) for option, file in step["inputs"].items()}
    options = [str(o) for o in step.get("options", [])]
    key = hashlib.sha256(json.dumps({
        "module": step["module"], "module_hash": _module_hash(step["module"]), "options": options,
        "output": step.get("output"), "inputs": {o: _hash(p) for o, p in inputs.items()},
    }, sort_keys=True).encode()).hexdigest()

    target, key_file = CACHE_DIR / name, CACHE_DIR / f"{name}.key"
    if target.is_file() and key_file.is_file() and key_file.read_text() == key:
        return target
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not modules.available():
        raise FileNotFoundError(f"{name} is built by bin/{step['module']}, which needs Linux x86-64 (the devcontainer)")
    argv = [str(modules.executable(step["module"]))]
    for option, file in inputs.items():
        argv += [f"--{option}", str(file)]
    argv += options
    output = step.get("output")
    argv += ["-o", str(target) if output is None else "/dev/null"]
    if output is not None:
        argv += [f"--{output}", str(target)]
    result = subprocess.run(argv, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"building {name} with {step['module']} failed:\n{result.stderr.decode()}")
    key_file.write_text(key)
    return target


def build_all() -> list[Path]:
    return [path(name) for name in steps()]

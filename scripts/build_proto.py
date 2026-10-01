"""Generate the ``osi3`` Python bindings from the OSI submodule.

The submodule tracks the upstream OSI branch ``feature/antenna-model``, which
contains the antenna extension on top of OSI v3.8.0. Use ``--upstream-only``
to build the plain upstream release it is based on instead.

    uv run python scripts/build_proto.py [--upstream-only]
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

from grpc_tools import protoc

ROOT = Path(__file__).resolve().parents[1]
OSI_DIR = ROOT / "third_party" / "open-simulation-interface"
UPSTREAM_REF = "v3.8.0"
BUILD_DIR = ROOT / "build" / "proto"
PROTO_PKG_DIR = BUILD_DIR / "osi3"
OUT_DIR = ROOT / "src"


def git_osi(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(OSI_DIR), *args], check=True, capture_output=True, text=True
    ).stdout


def read_sources(ref: str | None) -> dict[str, str]:
    """Proto sources, `VERSION` and the version template of the working tree or `ref`."""
    if not (OSI_DIR / "osi_common.proto").exists():
        sys.exit(f"OSI sources missing, run: git submodule update --init ({OSI_DIR})")
    if ref is None:
        names = [p.name for p in OSI_DIR.glob("*.proto")]
        read = lambda name: (OSI_DIR / name).read_text()  # noqa: E731
    else:
        names = [n for n in git_osi("ls-tree", "--name-only", ref).split() if n.endswith(".proto")]
        read = lambda name: git_osi("show", f"{ref}:{name}")  # noqa: E731
    return {name: read(name) for name in [*names, "VERSION", "osi_version.proto.in"]}


def write_protos(sources: dict[str, str]) -> None:
    version = dict(re.findall(r"(VERSION_\w+)\s*=\s*(\d+)", sources.pop("VERSION")))
    version_proto = sources.pop("osi_version.proto.in")
    for key, value in version.items():
        version_proto = version_proto.replace(f"@{key}@", value)
    sources["osi_version.proto"] = version_proto

    for name, text in sources.items():
        # Upstream protos import each other without a package prefix. Prefixing
        # them with "osi3/" makes the generated modules importable as `osi3.*`.
        text = re.sub(r'import "(osi_\w+\.proto)";', r'import "osi3/\1";', text)
        (PROTO_PKG_DIR / name).write_text(text)


def compile_protos() -> None:
    out_pkg = OUT_DIR / "osi3"
    shutil.rmtree(out_pkg, ignore_errors=True)
    out_pkg.mkdir(parents=True)
    grpc_include = Path(protoc.__file__).parent / "_proto"
    protos = sorted(str(p.relative_to(BUILD_DIR)) for p in PROTO_PKG_DIR.glob("*.proto"))
    result = protoc.main(
        [
            "protoc",
            f"-I{BUILD_DIR}",
            f"-I{grpc_include}",
            f"--python_out={OUT_DIR}",
            *protos,
        ]
    )
    if result != 0:
        sys.exit(f"protoc failed with exit code {result}")
    (out_pkg / "__init__.py").write_text("")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--upstream-only",
        action="store_true",
        help=f"build plain upstream OSI ({UPSTREAM_REF}) without the osi-hfss extension",
    )
    args = parser.parse_args()

    shutil.rmtree(BUILD_DIR, ignore_errors=True)
    PROTO_PKG_DIR.mkdir(parents=True)

    write_protos(read_sources(UPSTREAM_REF if args.upstream_only else None))
    compile_protos()

    variant = f"upstream {UPSTREAM_REF}" if args.upstream_only else "feature/antenna-model"
    print(f"Generated osi3 bindings ({variant}) in {OUT_DIR / 'osi3'}")


if __name__ == "__main__":
    main()

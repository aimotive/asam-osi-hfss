#!/usr/bin/env python3
"""Render a UML class diagram of a proto2 file (needs Graphviz `dot`).

Usage:
    python3 scripts/proto2uml.py osi_sensorview.proto
        -> osi_sensorview_uml.png
    python3 scripts/proto2uml.py osi_antenna.proto osi_radarcpi.proto
        -> one PNG per file
    python3 scripts/proto2uml.py osi_sensorview.proto --only RadarSensorView
        -> osi_sensorview_radarsensorview_uml.png
    python3 scripts/proto2uml.py osi_radarcpi.proto -o cpi.svg
        -> the format follows the extension (png, svg, pdf, ...; .dot keeps
           the Graphviz source, - prints it to stdout)

Messages and enums defined in the file become class boxes; nested types are
linked to their parent. Fields whose type is a message of the same file are
drawn as compositions. Types from other imported files are drawn as grey
boxes, except those from the files listed in --hide (shown as text only).
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

SCALARS = {
    "double",
    "float",
    "int32",
    "int64",
    "uint32",
    "uint64",
    "sint32",
    "sint64",
    "fixed32",
    "fixed64",
    "sfixed32",
    "sfixed64",
    "bool",
    "string",
    "bytes",
}
MULT = {"optional": "0..1", "repeated": "0..*", "required": "1"}


class Type:
    def __init__(self, kind, name, parent=None):
        self.kind = kind  # "message" | "enum"
        self.name = name
        self.parent = parent
        self.fields = []  # (label, type, name) or (name, value) for enums
        self.children = []

    @property
    def full(self):
        return f"{self.parent.full}.{self.name}" if self.parent else self.name


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def parse(path):
    """Return (list of top-level Types, list of imports)."""
    text = strip_comments(open(path).read())
    imports = re.findall(r'import\s+"([^"]+)"\s*;', text)
    tokens = re.findall(r"[{};]|[^{};]+", text)
    top, stack, prev = [], [], ""
    for tok in tokens:
        tok = " ".join(tok.split())
        if not tok:
            continue
        if tok not in "{};":
            prev = tok
        if tok == "}":
            stack.pop()
            continue
        if tok == "{":
            if not re.match(r"(message|enum|oneof)\s+\w+$", prev):
                stack.append(Type("skip", prev))  # e.g. extend / service blocks
            continue
        m = re.match(r"(message|enum|oneof)\s+(\w+)$", tok)
        if m:
            kind, name = m.groups()
            if kind == "oneof":  # flatten oneof fields into the parent
                stack.append(stack[-1])
                continue
            t = Type(kind, name, stack[-1] if stack else None)
            (stack[-1].children if stack else top).append(t)
            stack.append(t)
            continue
        if tok == ";" or not stack:
            continue
        cur = stack[-1]
        tok = re.sub(r"\[.*?\]", "", tok).strip()
        if cur.kind == "enum":
            m = re.match(r"(\w+)\s*=\s*(-?\d+)", tok)
            if m:
                cur.fields.append(m.groups())
        else:
            m = re.match(
                r"(optional|repeated|required)?\s*([\w.]+)\s+(\w+)\s*=\s*\d+", tok
            )
            if m:
                label, ftype, fname = m.groups()
                cur.fields.append((label or "optional", ftype, fname))
    return top, imports


def walk(types):
    for t in types:
        yield t
        yield from walk(t.children)


def resolve(ftype, scope, local):
    """Resolve a field type name to a local Type, searching enclosing scopes."""
    ftype = ftype.lstrip(".").removeprefix("osi3.")
    s = scope
    while s is not None:
        if f"{s.full}.{ftype}" in local:
            return local[f"{s.full}.{ftype}"]
        s = s.parent
    return local.get(ftype)


def enum_prefix(values):
    names = [n for n, _ in values]
    if len(names) < 2:
        return ""
    p = os.path.commonprefix(names)
    return p[: p.rfind("_") + 1] if "_" in p else ""


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def node_id(t):
    return t.full.replace(".", "__")


def class_box(t):
    if t.kind == "enum":
        prefix = enum_prefix(t.fields)
        rows = [f"{esc(n.removeprefix(prefix))} = {v}" for n, v in t.fields]
        head = f"<i>«enum»</i><br/><b>{esc(t.full.replace('.', '::'))}</b>"
        hb, bg = "#BFE0BF", "#EAF4EA"
    else:
        rows = [f"{esc(n)} : {esc(ft)} [{MULT[l]}]" for l, ft, n in t.fields]
        head = f"<b>{esc(t.full.replace('.', '::'))}</b>"
        hb, bg = "#F0D58C", "#FFF8DC"
    body = "<br/>".join(rows) or " "
    return (
        f'<<table border="1" cellborder="0" cellspacing="0" cellpadding="4" bgcolor="{bg}">'
        f'<tr><td bgcolor="{hb}">{head}</td></tr>'
        f'<tr><td align="left" balign="left">{body}</td></tr></table>>'
    )


def build_dot(proto, args):
    """Graphviz DOT source of the diagram of one proto file."""
    top, imports = parse(proto)
    local = {t.full: t for t in walk(top)}

    if args.only:
        todo = [local[n] for n in args.only.split(",")]
        keep = set()
        while todo:
            t = todo.pop()
            if t.full in keep:
                continue
            keep.add(t.full)
            todo += t.children
            if t.kind == "message":
                todo += [r for _, ft, _ in t.fields if (r := resolve(ft, t, local))]
        local = {n: t for n, t in local.items() if n in keep}

    # Map types of imported files to their file name.
    origin = {}
    base = os.path.dirname(os.path.abspath(proto))
    for imp in imports:
        p = os.path.join(base, imp)
        if not os.path.exists(p) and os.path.exists(p + ".in"):
            p += ".in"  # e.g. osi_version.proto is generated from a template
        if os.path.exists(p):
            for t in walk(parse(p)[0]):
                origin.setdefault(t.full, imp)
    hidden = set(filter(None, args.hide.split(",")))

    out = [
        f'digraph "{os.path.basename(proto)}" {{',
        f'  graph [rankdir={args.rankdir}, fontname="Helvetica", nodesep=0.4, ranksep=0.9,'
        f' label="osi3 :: {os.path.basename(proto)}", labelloc=t, fontsize=16];',
        '  node [shape=plain, fontname="Helvetica", fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=9, color="#444444"];',
    ]
    for t in local.values():
        out.append(f"  {node_id(t)} [label={class_box(t)}];")

    edges, externals = [], {}
    for t in local.values():
        for child in t.children:
            edges.append(
                f'  {node_id(t)} -> {node_id(child)} [dir=back, arrowtail=odot, style=solid, label=" nested"];'
            )
        if t.kind != "message":
            continue
        for label, ftype, fname in t.fields:
            if ftype in SCALARS:
                continue
            target = resolve(ftype, t, local)
            if target is not None:
                if target.kind == "enum" and target.parent is t:
                    continue  # already linked as nested
                arrow = "diamond" if target.kind == "message" else "none"
                edges.append(
                    f"  {node_id(t)} -> {node_id(target)} [dir=back, arrowtail={arrow},"
                    f' headlabel="{MULT[label]}", label=" {fname}"];'
                )
                continue
            name = ftype.removeprefix("osi3.")
            src = origin.get(name, "external")
            if src in hidden:
                continue
            ext = "ext__" + name.replace(".", "__")
            externals[ext] = (name, src)
            edges.append(
                f'  {node_id(t)} -> {ext} [arrowhead=vee, style=dotted, color="#888888",'
                f' fontcolor="#888888", headlabel="{MULT[label]}", label=" {fname}"];'
            )

    out.append(
        '  node [shape=box, style="rounded,dashed,filled", fillcolor="#EEEEEE",'
        ' color="#888888", fontcolor="#555555"];'
    )
    for ext, (name, src) in externals.items():
        out.append(
            f'  {ext} [label="«{src.removesuffix(".proto")}»\\n{name.replace(".", "::")}"];'
        )
    out += edges + ["}"]
    return "\n".join(out) + "\n"


def default_output(proto, only):
    stem = os.path.splitext(os.path.basename(proto))[0]
    if only:
        stem += "_" + "_".join(n.replace(".", "").lower() for n in only.split(","))
    return stem + "_uml.png"


def render(dot_source, output, dpi):
    if output == "-":
        sys.stdout.write(dot_source)
        return
    fmt = os.path.splitext(output)[1].lstrip(".").lower() or "png"
    if fmt == "dot":
        with open(output, "w") as f:
            f.write(dot_source)
    else:
        if shutil.which("dot") is None:
            sys.exit(
                "Graphviz 'dot' not found, install it (e.g. sudo apt install graphviz)"
                " or write the source with -o file.dot"
            )
        subprocess.run(
            ["dot", f"-T{fmt}", f"-Gdpi={dpi}", "-o", output],
            input=dot_source,
            text=True,
            check=True,
        )
    print(f"wrote {output}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("proto", nargs="+")
    ap.add_argument(
        "-o",
        "--output",
        help="output file, format from the extension (default <proto>_uml.png in the"
        " current directory, only with a single proto); - prints DOT to stdout",
    )
    ap.add_argument("--dpi", type=int, default=150, help="resolution of raster output")
    ap.add_argument(
        "--hide",
        default="osi_common.proto,osi_version.proto",
        help="comma separated imported files whose types are not drawn as boxes",
    )
    ap.add_argument("--rankdir", default="LR", choices=["LR", "TB"])
    ap.add_argument(
        "--only",
        default="",
        help="comma separated messages to draw, with their nested and referenced local types",
    )
    args = ap.parse_args()
    if args.output and len(args.proto) > 1:
        ap.error("-o/--output needs a single proto file")

    for proto in args.proto:
        render(
            build_dot(proto, args),
            args.output or default_output(proto, args.only),
            args.dpi,
        )


if __name__ == "__main__":
    sys.exit(main())

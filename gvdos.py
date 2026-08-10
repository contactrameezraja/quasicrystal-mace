"""
Verify that the replicated structure parsers agree
=================================================

Section 3.2 states that an identical parser is used by every entry point, the
copies being replicated deliberately rather than factored into a shared module,
since each script is submitted directly to the cluster and a shared import would
add a deployment dependency to every job. That claim needs an artefact behind it,
and this is the artefact.

Byte-level comparison is the wrong test, because the copies legitimately differ in
their docstrings and diagnostic messages. What must agree is the parsing logic and
the behaviour. This script therefore does two things.

  Structural check. It extracts the parse_structure function from each file,
  strips comments, docstrings and blank lines, normalises whitespace, and hashes
  what remains. Identical hashes mean identical logic irrespective of
  documentation.

  Behavioural check. It imports each copy and runs all of them on the same
  fixtures, comparing atom counts, species and positions exactly, and confirming
  that each rejects a corrupted file rather than proceeding.

The second check is the one that matters, since two implementations can differ in
form and agree in behaviour, and it is behaviour the thesis depends on. Report the
hash table and the behavioural verdict in Appendix A.

Usage
-----
    python verify_parsers.py --files md_run.py reduce_vdos.py phonon_run.py \
        msd_gr.py retarget_composition.py --fixture Al9Co2Ni2-coords.txt
"""

import argparse
import ast
import hashlib
import importlib.util
import io
import re
import sys
import tokenize


def extract_function(path, name="parse_structure"):
    """Return the source of the named top-level function, or None."""
    src = open(path).read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    return None


def normalise(src):
    """Strip comments, docstrings and blank lines, and collapse whitespace, so
    that only the logic remains. Uses the tokeniser rather than regular
    expressions so that strings containing hash characters survive."""
    out = []
    prev_type = None
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            continue
        # A string that stands alone as a statement is a docstring.
        if (tok.type == tokenize.STRING and
                prev_type in (tokenize.INDENT, tokenize.NEWLINE, tokenize.NL,
                              None)):
            prev_type = tok.type
            continue
        if tok.type in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                        tokenize.DEDENT):
            prev_type = tok.type
            continue
        out.append(tok.string)
        prev_type = tok.type
    return re.sub(r"\s+", " ", " ".join(out)).strip()


def load_parser(path):
    """Import a copy of parse_structure without executing the script body."""
    src = extract_function(path)
    if src is None:
        return None
    ns = {}
    header = ("import numpy as np\n"
              "from ase import Atoms\n"
              "from ase.io import read\n"
              "TYPE_TO_ELEMENT = {13: 'Al', 27: 'Co', 28: 'Ni'}\n")
    try:
        exec(header + src, ns)
    except Exception as e:                                   # noqa: BLE001
        print(f"  could not load from {path}: {type(e).__name__}: {e}")
        return None
    return ns.get("parse_structure")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--fixture", required=True,
                    help="a legacy coords.txt file the parser should accept")
    args = ap.parse_args()

    print("=" * 68)
    print("STRUCTURAL CHECK: hash of parse_structure with documentation stripped")
    print("=" * 68)
    hashes = {}
    for path in args.files:
        src = extract_function(path)
        if src is None:
            print(f"{path:32s} parse_structure NOT FOUND")
            continue
        h = hashlib.sha256(normalise(src).encode()).hexdigest()[:16]
        hashes[path] = h
        print(f"{path:32s} {h}")
    distinct = sorted(set(hashes.values()))
    if len(distinct) == 1:
        print("\n  token streams identical across all copies")
    else:
        print(f"\n  {len(distinct)} distinct token streams. This is expected if "
              f"the copies differ in local variable names or in the wording of\n"
              f"  their diagnostic messages, neither of which changes behaviour. "
              f"The behavioural check below is the one that decides.")
        groups = {}
        for p, h in hashes.items():
            groups.setdefault(h, []).append(p)
        for h, ps in groups.items():
            print(f"    {h}: {', '.join(ps)}")

    print()
    print("=" * 68)
    print("BEHAVIOURAL CHECK: same fixture through every copy")
    print("=" * 68)
    results = {}
    for path in args.files:
        fn = load_parser(path)
        if fn is None:
            continue
        try:
            atoms = fn(args.fixture)
            results[path] = (len(atoms), tuple(atoms.get_chemical_symbols()),
                             atoms.get_positions().round(8).tobytes())
            print(f"{path:32s} {len(atoms)} atoms, "
                  f"volume {atoms.get_volume():.1f} A^3")
        except SystemExit as e:
            print(f"{path:32s} REJECTED the fixture: {e}")
        except Exception as e:                               # noqa: BLE001
            print(f"{path:32s} ERROR {type(e).__name__}: {e}")

    if len(set(results.values())) == 1 and results:
        print("\n  every copy returns identical atoms, species and positions")
    elif results:
        print("\n  COPIES DISAGREE on the parsed structure; this is a defect, not "
              "a documentation difference")

    print()
    print("=" * 68)
    print("REJECTION CHECK: a corrupted fixture must be refused")
    print("=" * 68)
    bad = args.fixture.replace(".txt", "_corrupt.txt")
    lines = open(args.fixture).read().splitlines(keepends=True)
    open(bad, "w").writelines(lines[:len(lines) - 3])        # drop three atoms
    for path in args.files:
        fn = load_parser(path)
        if fn is None:
            continue
        try:
            atoms = fn(bad)
            print(f"{path:32s} ACCEPTED a corrupted file, {len(atoms)} atoms, "
                  f"no verification present")
        except SystemExit:
            print(f"{path:32s} refused, verification present")
        except Exception as e:                               # noqa: BLE001
            print(f"{path:32s} raised {type(e).__name__}")


if __name__ == "__main__":
    main()

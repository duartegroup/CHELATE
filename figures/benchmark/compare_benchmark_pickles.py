"""Compare two benchmark_results.pkl files and list every differing leaf.

Usage: python compare_benchmark_pickles.py A.pkl B.pkl [--tol 1e-9]
Keys present in only one file are reported but not counted as differences.
"""

import argparse
import pickle

import numpy as np


def walk(a, b, path, out, tol):
    if a is None or b is None:
        if a is None and b is None:
            out["same"] += 1
        else:
            out["diff"].append(f"{path}: {type(a).__name__} vs {type(b).__name__}")
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out["only"].append(f"{path}/{k} only in {'B' if k not in a else 'A'}")
                continue
            walk(a[k], b[k], f"{path}/{k}", out, tol)
        return
    if isinstance(a, list) and a and isinstance(a[0], dict):
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{x.get('name', i)}]", out, tol)
        if len(a) != len(b):
            out["diff"].append(f"{path}: length {len(a)} vs {len(b)}")
        return
    if isinstance(a, str) or (isinstance(a, list) and a and isinstance(a[0], str)):
        if a != b:
            out["diff"].append(f"{path}: {a!r} vs {b!r}")
        return
    try:
        x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    except (TypeError, ValueError):
        # Cohort provenance contains nested (system_id, ligand_id) string pairs.
        x, y = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
        if x.shape != y.shape:
            out["diff"].append(f"{path}: shape {x.shape} vs {y.shape}")
        elif not np.array_equal(x, y):
            out["diff"].append(f"{path}: nonnumeric values differ")
        else:
            out["same"] += 1
        return
    if x.shape != y.shape:
        out["diff"].append(f"{path}: shape {x.shape} vs {y.shape}")
    elif x.size and not np.allclose(x, y, atol=tol, rtol=0, equal_nan=True):
        out["diff"].append(f"{path}: max abs diff {np.nanmax(np.abs(x - y)):.3g}")
    else:
        out["same"] += 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--tol", type=float, default=1e-9)
    args = p.parse_args()
    A, B = (pickle.load(open(f, "rb")) for f in (args.a, args.b))
    out = {"diff": [], "only": [], "same": 0}
    walk(A, B, "", out, args.tol)
    print(f"identical leaves: {out['same']}, differing leaves: {len(out['diff'])}")
    for line in out["diff"]:
        print("DIFF", line)
    for line in out["only"]:
        print("ONLY", line)

#!/usr/bin/env python3
"""
overlap_diagnostic_v2.py
========================
Robust successor to overlap_diagnostic.py. Instead of comparing pair_id *strings*
(which are ambiguous for large protein indices and sensitive to concat order),
this version:

  1. Decodes each released pair_id string into its underlying (idx1, idx2) tuple
     by brute-force searching the proteinList range, trying both concat orders.
  2. Compares as unordered frozenset({i, j}) edge tuples — invariant to both
     concat direction and edge orientation.

Also reports *why* a pair_id failed to decode (ambiguous / no valid split),
which diagnoses whether the discrepancy is a format bug or a real data mismatch.
"""

import os
from collections import Counter

import numpy as np


RAW_ROOT       = "/home/membio8/Methods_local/S-VGAE/data"
PROCESSED_ROOT = "/home/membio8/Methods_local/data"
SEQVEC_ROOT    = "/home/membio8/Methods_local/seqvec_files"

DATASETS = [
    ("Hprd",       "ppi",        "hprd_seqvec_dict.npy"),
    ("C.elegan",   "c.elegan",   "C.elegan_seqvec_dict.npy"),
    ("Drosophila", "drosophila", None),
    ("E.coli",     "e.coli",     None),
]


def load_protein_list(path):
    ids = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            ids.append(parts[1] if len(parts) > 1 else parts[0])
    return ids


def load_edges(pos_path, neg_path):
    edges = []
    for path, is_pos in [(pos_path, True), (neg_path, False)]:
        with open(path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    edges.append((int(parts[0]), int(parts[1]), is_pos))
    return edges


def load_released_pair_ids(node_path):
    ids = []
    with open(node_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            first = line.split("\t", 1)[0] if "\t" in line else line.split(None, 1)[0]
            ids.append(first)
    return ids


def decode_pair_id(pair_id_str, n_proteins):
    """Try all possible splits of the digit string into (a, b) where both
    a, b are valid protein indices (< n_proteins). Return list of possible
    frozenset({a, b}) decodings (could be 0, 1, or more)."""
    candidates = set()
    s = pair_id_str
    for split in range(1, len(s)):
        left, right = s[:split], s[split:]
        # reject leading zeros except for the single digit "0"
        if (len(left) > 1 and left[0] == "0") or (len(right) > 1 and right[0] == "0"):
            continue
        a, b = int(left), int(right)
        if a < n_proteins and b < n_proteins:
            candidates.add(frozenset({a, b}))
    return candidates


def diagnose(raw_name, proc_name, seqvec_name):
    print(f"\n{'='*72}")
    print(f"  Dataset: {raw_name}  (processed dir: '{proc_name}')")
    print(f"{'='*72}")

    raw_dir  = os.path.join(RAW_ROOT, raw_name)
    proc_dir = os.path.join(PROCESSED_ROOT, proc_name)

    plist_path = os.path.join(raw_dir, "proteinList.txt")
    pos_path   = os.path.join(raw_dir, "PositiveEdges.txt")
    neg_path   = os.path.join(raw_dir, "NegativeEdges.txt")
    node_path  = os.path.join(proc_dir, "node")

    for p in (plist_path, pos_path, neg_path, node_path):
        if not os.path.exists(p):
            print(f"  [skip] missing: {p}")
            return

    proteins = load_protein_list(plist_path)
    n = len(proteins)
    print(f"  proteins: {n}")

    edges = load_edges(pos_path, neg_path)
    raw_edge_set = set(frozenset({i, j}) for (i, j, _) in edges)
    print(f"  raw edges: {len(edges)} ({len(raw_edge_set)} unique unordered)")

    released_ids = load_released_pair_ids(node_path)
    print(f"  released node file: {len(released_ids)} lines")

    # --- decode released pair_ids into edge tuples ---
    decoded_edges = set()        # frozenset({i,j}) for each unambiguously-decoded id
    ambiguous = 0
    unresolvable = 0
    for pid in released_ids:
        cands = decode_pair_id(pid, n)
        if len(cands) == 0:
            unresolvable += 1
        elif len(cands) == 1:
            decoded_edges.add(next(iter(cands)))
        else:
            # ambiguous — still add all candidates to the "possible" set but count it
            ambiguous += 1
            # for the comparison, be charitable: include every candidate
            decoded_edges.update(cands)

    print(f"  decoded: {len(decoded_edges)} unique edges from released ids")
    print(f"    ambiguous ids (multiple valid splits): {ambiguous}")
    print(f"    unresolvable ids (no valid split):     {unresolvable}")

    # --- overlap of edge sets ---
    inter = decoded_edges & raw_edge_set
    only_rel = decoded_edges - raw_edge_set
    only_raw = raw_edge_set - decoded_edges
    pct = 100 * len(inter) / max(1, len(decoded_edges))
    print(f"  edge-set overlap:")
    print(f"    released ∩ raw : {len(inter)} ({pct:.1f}% of decoded)")
    print(f"    only in released (can't find in raw): {len(only_rel)}")
    print(f"    only in raw (not in released):        {len(only_raw)}")

    # --- sanity: do labels match? ---
    # Build a label map from raw, then check released labels (last column) match.
    raw_pos = set(frozenset({i, j}) for (i, j, p) in edges if p)
    raw_neg = set(frozenset({i, j}) for (i, j, p) in edges if not p)
    # (We'd need to re-read the released file with labels to cross-check — skipping for brevity.)

    # --- first few examples of released pair_ids and how they decode ---
    print(f"  first 5 released ids → decodings:")
    for pid in released_ids[:5]:
        cands = decode_pair_id(pid, n)
        matches = [c for c in cands if c in raw_edge_set]
        print(f"    '{pid}'  candidates={[tuple(sorted(c)) for c in cands]}  in_raw={[tuple(sorted(c)) for c in matches]}")


def main():
    print("Robust overlap diagnostic (edge-set based)")
    for raw, proc, sv in DATASETS:
        try:
            diagnose(raw, proc, sv)
        except Exception as e:
            print(f"\n[ERROR] {raw}: {type(e).__name__}: {e}")
    print("\nDone.")


if __name__ == "__main__":
    main()
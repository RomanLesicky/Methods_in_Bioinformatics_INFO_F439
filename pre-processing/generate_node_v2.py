#!/usr/bin/env python3
"""
Dataset-aware node-file generator for Graph-BERT / Graph-BERT-ESM2.

Key fixes:
1. Writes into the current Methods repo's Node_creation directory.
2. Resolves the embedding file from --embedder instead of only changing output name.
3. Filters missing-embedding edges BEFORE shuffling.
4. Shuffles deterministically for reproducible row order.
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# This file lives in: Methods/pre-processing/generate_node_v2.py
# So repo root is one level up from this script's directory.
REPO_ROOT = Path(__file__).resolve().parents[1]

# If your raw S-VGAE data live elsewhere, you can override with:
# export PPI_RAW_ROOT=/your/path/to/S-VGAE/data
RAW_ROOT = Path(os.environ.get("PPI_RAW_ROOT", REPO_ROOT / "S-VGAE" / "data"))

SEQVEC_ROOT = REPO_ROOT / "seqvec_files"
ESM_ROOT    = REPO_ROOT / "esm_files"
OUT_ROOT    = REPO_ROOT / "Node_creation"


# ---------------------------------------------------------------------------
# Per-dataset configuration
# ---------------------------------------------------------------------------
DATASETS = {
    # dataset_name : (raw_subdir, n_pos_keep)
    "hprd":       ("Hprd",       5000),
    "c.elegan":   ("C.elegan",   None),
    "e.coli":     ("E.coli",     None),
    "drosophila": ("Drosophila", None),
    "human":      ("Human",      None),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def resolve_embed_path(raw_subdir: str, embedder: str, override: str | None) -> Path:
    """
    Resolve the embedding dictionary path from the dataset name + embedder tag.

    Supported short tags:
      - seqvec
      - esm2_650M  -> esm2_t33_650M_UR50D
      - esm2_3B    -> esm2_t36_3B_UR50D
    """
    if override is not None:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(f"--embed-file does not exist: {p}")
        return p

    # Map your short CLI tags to the actual filename fragments on disk
    embedder_alias = {
        "seqvec": "seqvec",
        "esm2_650M": "esm2_t33_650M_UR50D",
        "esm2_3B": "esm2_t36_3B_UR50D",
    }

    if embedder not in embedder_alias:
        raise ValueError(
            f"Unsupported embedder: {embedder}. "
            f"Expected one of {list(embedder_alias.keys())}"
        )

    embed_token = embedder_alias[embedder]

    # seqvec lives in seqvec_files, ESM lives in esm_files
    if embedder == "seqvec":
        search_roots = [SEQVEC_ROOT]
    else:
        search_roots = [ESM_ROOT]

    # Try a few exact candidates first
    candidates = []
    for root in search_roots:
        candidates.extend([
            root / f"{raw_subdir}_{embed_token}_dict.npy",
            root / f"{raw_subdir.lower()}_{embed_token}_dict.npy",
        ])

    for p in candidates:
        if p.exists():
            return p

    # Then try glob fallback
    patterns = [
        f"{raw_subdir}_{embed_token}_dict.npy",
        f"{raw_subdir.lower()}_{embed_token}_dict.npy",
        f"*{raw_subdir}*{embed_token}*_dict.npy",
        f"*{raw_subdir.lower()}*{embed_token}*_dict.npy",
    ]

    matches = []
    for root in search_roots:
        if root.exists():
            for pattern in patterns:
                matches.extend(sorted(root.glob(pattern)))

    # Deduplicate while preserving order
    seen = set()
    unique_matches = []
    for m in matches:
        if m not in seen:
            unique_matches.append(m)
            seen.add(m)

    if len(unique_matches) == 1:
        return unique_matches[0]

    searched = [str(p) for p in candidates]
    raise FileNotFoundError(
        "Could not resolve embedding dictionary automatically.\n"
        f"dataset={raw_subdir}, embedder={embedder}, mapped_token={embed_token}\n"
        f"Tried exact candidates:\n  - " + "\n  - ".join(searched) + "\n"
        f"Also searched glob patterns in: {', '.join(str(r) for r in search_roots)}\n"
        "Use --embed-file explicitly if needed."
    )


def protein_key(prot_lines, idx):
    return prot_lines[idx].strip().split("\t")[1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    ap.add_argument(
        "--embedder",
        default="seqvec",
        choices=["seqvec", "esm2_650M", "esm2_3B"],
        help="Embedding source tag used both for lookup and output filename."
    )
    ap.add_argument(
        "--embed-file",
        default=None,
        help="Optional explicit .npy embedding dictionary path."
    )
    args = ap.parse_args()

    raw_subdir, n_pos_keep = DATASETS[args.dataset]
    raw_dir = RAW_ROOT / raw_subdir
    embed_path = resolve_embed_path(raw_subdir, args.embedder, args.embed_file)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Repo root  : {REPO_ROOT}")
    print(f"Dataset    : {args.dataset}")
    print(f"Raw dir    : {raw_dir}")
    print(f"Embedder   : {args.embedder}")
    print(f"Embed file : {embed_path}")
    print(f"Output dir : {OUT_ROOT}")

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}")

    data = np.load(embed_path, allow_pickle=True).tolist()
    print(f"Loaded {len(data)} embeddings (sample dim={len(next(iter(data.values())))})")

    with open(raw_dir / "proteinList.txt") as f:
        prot_lines = f.readlines()

    pos_edges = []
    with open(raw_dir / "PositiveEdges.txt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                pos_edges.append((int(parts[0]), int(parts[1])))

    neg_edges = []
    with open(raw_dir / "NegativeEdges.txt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                neg_edges.append((int(parts[0]), int(parts[1])))

    print(f"Positive edges: {len(pos_edges)}")
    print(f"Negative edges: {len(neg_edges)}")

    if n_pos_keep is not None and len(pos_edges) > n_pos_keep:
        print(f"Applying filter: keep first {n_pos_keep} positives + all negatives")
        edge_list = (
            [(i, j, True) for (i, j) in pos_edges[:n_pos_keep]] +
            [(i, j, False) for (i, j) in neg_edges]
        )
    else:
        edge_list = (
            [(i, j, True) for (i, j) in pos_edges] +
            [(i, j, False) for (i, j) in neg_edges]
        )

    # First filter to only edges for which both protein embeddings exist.
    missing_proteins = set()
    kept_edges = []
    n_skipped_missing = 0

    for prot1, prot2, is_pos in edge_list:
        if prot1 >= len(prot_lines) or prot2 >= len(prot_lines):
            n_skipped_missing += 1
            continue

        key1 = protein_key(prot_lines, prot1)
        key2 = protein_key(prot_lines, prot2)

        v1 = data.get(key1)
        v2 = data.get(key2)

        if v1 is None or v2 is None:
            if v1 is None:
                missing_proteins.add(key1)
            if v2 is None:
                missing_proteins.add(key2)
            n_skipped_missing += 1
            continue

        kept_edges.append((prot1, prot2, is_pos))

    # Deterministic shuffle of the kept edges
    rng = random.Random(42)
    rng.shuffle(kept_edges)

    n_pos_kept = sum(1 for _, _, is_pos in kept_edges if is_pos)
    n_neg_kept = len(kept_edges) - n_pos_kept

    print(f"Kept edges: {len(kept_edges)} ({n_pos_kept} pos, {n_neg_kept} neg)")
    print(f"Skipped (missing embedding or oob): {n_skipped_missing}")
    if missing_proteins:
        print(f"{len(missing_proteins)} unique proteins had no embedding")

    out_name = f"Node_{raw_subdir}_{args.embedder}.txt"
    out_path = OUT_ROOT / out_name

    with open(out_path, "w") as f:
        for prot1, prot2, is_pos in kept_edges:
            key1 = protein_key(prot_lines, prot1)
            key2 = protein_key(prot_lines, prot2)

            v1 = data[key1]
            v2 = data[key2]

            feat = list(v1) + list(v2)
            pair_id = str(prot2) + str(prot1)   # preserve original generate_node.py convention
            label = "Positive" if is_pos else "Negative"

            f.write(pair_id + "\t" + "\t".join(str(x) for x in feat) + "\t" + label + "\n")

    print(f"\nWrote {len(kept_edges)} nodes → {out_path}")
    print(f"  Positive: {n_pos_kept}")
    print(f"  Negative: {n_neg_kept}")
    print(f"  File size: {out_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
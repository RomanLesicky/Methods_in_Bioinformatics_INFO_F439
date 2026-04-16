#!/usr/bin/env python3
"""
regenerate_seqvec_dicts.py
==========================
Rebuild the per-protein SeqVec embedding dictionaries that the PPI Graph-BERT
pipeline expects at ``seqvec_files/<dataset>_seqvec_dict.npy``.

The original .npy files on the authors' GitHub are stored via Git LFS and the
remote objects have been deleted (404 on ``git lfs pull``). We rebuild them
from the raw ``proteinList.txt`` + sequence file pairs in the S-VGAE data.

Embedding recipe mirrors the original ``embedding.py`` exactly:

    emb = SeqVecEmbedder().embed(sequence)                 # (3, L, 1024)
    per_protein = torch.tensor(emb).sum(dim=0).mean(dim=0) # (1024,)

Output dict schema:  { uniprot_id: np.ndarray shape (1024,) dtype float32 }

Requirements
------------
    python -m venv ~/venvs/bio_emb          # fresh venv strongly recommended
    source ~/venvs/bio_emb/bin/activate
    pip install bio-embeddings[seqvec]

First run downloads ~400 MB of SeqVec weights to ~/.cache/bio_embeddings.

Runtime on an RTX 6000 Ada: roughly 5-10 min per 1000 proteins, so:
    HPRD       9463 proteins  ~60 min
    Human      ~XX   proteins
    Drosophila 5624 proteins  ~35 min
    E.coli     1528 proteins  ~10 min
    C.elegan   1734 proteins  ~10 min
"""

import os

_N = "8"
os.environ["OMP_NUM_THREADS"]         = _N   
os.environ["MKL_NUM_THREADS"]         = _N   
os.environ["OPENBLAS_NUM_THREADS"]    = _N   
os.environ["NUMEXPR_NUM_THREADS"]     = _N   

import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_ROOT = "/home/membio8/Methods_local/S-VGAE/data"
OUT_ROOT = "/home/membio8/Methods_local/seqvec_files"

# (raw subdir, sequence filename, output dict filename)
# Output names preserve the original repo's naming for HPRD and C.elegan,
# and extend the pattern for the three datasets that never had a dict shipped.
DATASETS = [
    ("C.elegan",   "sequenceList.txt",  "C.elegan_seqvec_dict.npy"),
    ("E.coli",     "sequenceList.txt",  "e.coli_seqvec_dict.npy"),
    ("Drosophila", "sequenceList.txt",  "drosophila_seqvec_dict.npy"),
    ("Hprd",       "sequence.txt",      "hprd_seqvec_dict.npy"),
    ("Human",      "sequenceList.txt",  "human_seqvec_dict.npy"),
]
# Ordered small→large so you get quick confirmation the pipeline works
# before committing to the hour-long HPRD run.


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------
def load_ids_and_sequences(raw_dir: str, seq_filename: str):
    """Parallel read of proteinList.txt and the sequence file.

    The two files are expected to be parallel (line N of proteinList gives the
    ID for line N of sequence file). In the S-VGAE distribution for some
    species (e.g. C.elegan, E.coli, Drosophila), proteinList.txt contains extra
    trailing entries that are never referenced by any edge and have no
    corresponding sequence. We silently truncate to the overlap.

    Returns list of (uniprot_id, sequence) tuples, in file order, length
    = min(len(proteinList), len(sequences)).
    """
    plist_path = os.path.join(raw_dir, "proteinList.txt")
    seq_path   = os.path.join(raw_dir, seq_filename)

    ids = []
    with open(plist_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            ids.append(parts[1] if len(parts) > 1 else parts[0])

    seqs = []
    with open(seq_path) as f:
        for line in f:
            seqs.append(line.strip())

    n = min(len(ids), len(seqs))
    if len(ids) != len(seqs):
        print(f"  note: proteinList has {len(ids)} entries, "
              f"sequence file has {len(seqs)}. Using first {n} as the parallel overlap.")

    return list(zip(ids[:n], seqs[:n]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    try:
        import torch
        from bio_embeddings.embed import SeqVecEmbedder
    except ImportError as e:
        print(f"ERROR: {e}")
        print("Install with: pip install bio-embeddings[seqvec]")
        print("(Strongly recommend doing this in a fresh venv — bio_embeddings pins old deps.)")
        sys.exit(1)

    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SeqVec embedder on device='{device}' ...")
    embedder = SeqVecEmbedder(device=device)
    print("  SeqVec loaded.\n")

    os.makedirs(OUT_ROOT, exist_ok=True)

    for (raw_name, seq_filename, out_name) in DATASETS:
        raw_dir  = os.path.join(RAW_ROOT, raw_name)
        out_path = os.path.join(OUT_ROOT, out_name)

        print(f"{'='*72}")
        print(f"  {raw_name}")
        print(f"{'='*72}")

        if not os.path.isdir(raw_dir):
            print(f"  [skip] raw dir not found: {raw_dir}")
            continue

        if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
            print(f"  [skip] output already exists ({os.path.getsize(out_path)/1e6:.1f} MB): {out_path}")
            print(f"         delete it if you want to regenerate.")
            continue

        try:
            pairs = load_ids_and_sequences(raw_dir, seq_filename)
        except (FileNotFoundError, ValueError) as e:
            print(f"  [skip] {e}")
            continue

        # Sanity print — check the format looks right before committing to hours of compute
        sid, sseq = pairs[0]
        print(f"  {len(pairs)} protein entries to embed")
        print(f"  sample:  id='{sid}'  seq='{sseq[:60]}{'...' if len(sseq) > 60 else ''}'  (len={len(sseq)})")

        if not sseq or not sseq[0].isalpha():
            print(f"  WARNING: first sequence looks wrong. Aborting this dataset.")
            continue

        out_dict = {}
        failed = []
        t_start = time.time()

        for i, (uid, seq) in enumerate(pairs):
            if not seq:
                failed.append((uid, "empty sequence"))
                continue
            try:
                emb = embedder.embed(seq)                                  # (3, L, 1024) numpy
                vec = torch.from_numpy(np.asarray(emb)).sum(dim=0).mean(dim=0)  # (1024,)
                out_dict[uid] = vec.cpu().numpy().astype(np.float32)
            except Exception as e:
                failed.append((uid, str(e)[:80]))

            if (i + 1) % 100 == 0 or (i + 1) == len(pairs):
                elapsed = time.time() - t_start
                rate = (i + 1) / max(elapsed, 1e-6)
                remaining = (len(pairs) - (i + 1)) / max(rate, 1e-6)
                print(f"    [{i+1}/{len(pairs)}]  {rate:.1f} prot/s  "
                      f"elapsed {elapsed/60:.1f} min  eta {remaining/60:.1f} min")

        # Save — format mirrors how generate_node.py loads it:
        #   np.load(path, allow_pickle=True).tolist()  -> dict
        np.save(out_path, out_dict, allow_pickle=True)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"\n  saved {len(out_dict)} embeddings ({size_mb:.1f} MB) → {out_path}")
        if failed:
            print(f"  {len(failed)} proteins failed to embed:")
            for uid, err in failed[:10]:
                print(f"    {uid}: {err}")
            if len(failed) > 10:
                print(f"    ... and {len(failed) - 10} more")

    print("\nAll datasets processed.")


if __name__ == "__main__":
    main()

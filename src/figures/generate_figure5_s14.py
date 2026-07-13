#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OncoPep Step 9 — PLOS Computational Biology Figure 5 rescue script.

Scientific role
---------------
Step 9 focuses on contextual similarity support and internal diversity of the
final OncoPep candidate panel. It intentionally avoids the Step 8 selection
and prioritization-score panels.

Main figure
-----------
Figure 5. Contextual similarity and compositional support of the final
OncoPep panel.
  A. NN similarity context
  B. Similarity summary
  C. Residue-category context

Supplementary figure
--------------------
Supplementary Figure S14. Additional sequence-composition context.
  A. Amino-acid enrichment
  B. Top enriched 3-mers

Removed from plotted supplementary figure
-----------------------------------------
The pairwise final-panel similarity heatmap is retained in source data and
reports, but is no longer plotted as S14C to keep the supplement compact.

Design decisions
----------------
- Rescues the scientifically stronger old Step 9 similarity/composition logic.
- Removes old Step 9 selection-audit and score-shift panels because they now
  duplicate Step 8/Figure 4.
- Uses uppercase panel labels, OncoPep/PLOS palette, source-data exports,
  manifest, README, requirements, code snapshot, and readiness reporting.
- Implements the approved panel swap: residue-category context is used as
  main Figure 5C.
- Generates Supplementary Figure S14 as a two-panel composition figure by
  default whenever amino-acid and k-mer enrichment data can be loaded or computed.
- Centers Supplementary Figure S14 panel titles and assigns a unique restrained
  PLOS-compatible color to each top 3-mer bar.
- Retains pairwise final-panel similarity as source data/report rather than a
  plotted supplementary panel.
- Avoids default all-1.00 contextual-support fallback panels.

Ready-to-run examples
---------------------
python OncoPep_step9_PLOS_contextual_similarity_diversity_final.py \
  --step9-root /home/data3/Moe/nature_computational_peponco/PLOS/plos_comp/step_09 \
  --project-root /home/data3/Moe/nature_computational_peponco \
  --legacy-step9-root /home/data3/Moe/nature_computational2/step9_v1

For explicit paths:
python OncoPep_step9_PLOS_contextual_similarity_diversity_final.py \
  --final-panel-file /path/to/table_s9_9_final_panel_with_all_step9_annotations.csv \
  --reference-file /path/to/step1_retained_dataset.csv \
  --pairwise-file /path/to/table_s9_7_final_panel_pairwise_similarity.csv \
  --aa-enrichment-file /path/to/table_s9_5_amino_acid_enrichment_vs_reference.csv \
  --kmer-enrichment-file /path/to/table_s9_6_kmer_enrichment_vs_reference.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import traceback
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

SCRIPT_VERSION = "v6_2026_07_08_step9_s14_center_titles_unique_kmer_colors"

DEFAULT_STEP9_ROOT = "/home/data3/Moe/nature_computational_peponco/PLOS/plos_comp/step_09"
DEFAULT_PROJECT_ROOT = "/home/data3/Moe/nature_computational_peponco"
DEFAULT_LEGACY_STEP9_ROOT = "/home/data3/Moe/nature_computational2/step9_v1"
DEFAULT_LEGACY_REFERENCE = "/home/data3/Moe/nature_computational2/step1/tables/step1_retained_dataset.csv"

SAFE_TABLE_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".pq"}
BLOCKED_EXTENSIONS = {".json", ".png", ".jpg", ".jpeg", ".pdf", ".tif", ".tiff", ".py", ".ipynb"}
BLOCKED_PATH_KEYWORDS = [
    "/run/user/",
    "/jupyter/runtime/",
    ".ipynb_checkpoints",
    "__pycache__",
    "/.git/",
    "kernel-",
]

PLOS = {
    "blue": "#1F95B8",
    "mint": "#A8D3B2",
    "coral": "#DD705D",
    "brown": "#B67D4E",
    "charcoal": "#6A5E61",
    "dark": "#333333",
    "grid": "#EAEAEA",
    "light": "#D9D9D9",
    "white": "#FFFFFF",
    "edge": "#B8B8B8",
    "pale": "#F5F7F7",
}

AA_SET = set("ACDEFGHIKLMNPQRSTVWY")
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_HYDROPHOBIC = set("AILMFWVYC")
AA_AROMATIC = set("FYW")
AA_POSITIVE = set("KRH")
AA_NEGATIVE = set("DE")
AA_POLAR = set("STNQCYWHKRD")

KD_HYDROPATHY = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}


@dataclass
class CheckResult:
    name: str
    status: str  # PASS/WARN/FAIL
    message: str


@dataclass
class OutputDirs:
    root: Path
    main_figure: Path
    supplementary_figures: Path
    source_data: Path
    reports: Path
    code: Path


def is_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ in {"ZMQInteractiveShell", "TerminalInteractiveShell"}
    except Exception:
        return False


def clean_argv(argv: Optional[Sequence[str]] = None) -> List[str]:
    if argv is None:
        argv = sys.argv[1:]
    out: List[str] = []
    skip = False
    for i, token in enumerate(argv):
        if skip:
            skip = False
            continue
        s = str(token)
        if s == "-f":
            if i + 1 < len(argv):
                skip = True
            continue
        if any(k in s for k in BLOCKED_PATH_KEYWORDS):
            continue
        if s.endswith(".json") and ("kernel-" in s or "/runtime/" in s):
            continue
        out.append(s)
    return out


def ensure_dirs(root: Path) -> OutputDirs:
    dirs = OutputDirs(
        root=root,
        main_figure=root / "main_figure",
        supplementary_figures=root / "supplementary_figures",
        source_data=root / "source_data",
        reports=root / "reports",
        code=root / "code",
    )
    for d in asdict(dirs).values():
        if isinstance(d, Path):
            d.mkdir(parents=True, exist_ok=True)
    return dirs


def is_blocked_path(path: Path) -> bool:
    p = str(path).replace("\\", "/").lower()
    return any(k.lower() in p for k in BLOCKED_PATH_KEYWORDS)


def is_allowed_table_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    return suffix in SAFE_TABLE_EXTENSIONS and suffix not in BLOCKED_EXTENSIONS and not is_blocked_path(path)


def safe_roots(*roots: Optional[str]) -> List[Path]:
    seen = set()
    out: List[Path] = []
    for r in roots:
        if not r:
            continue
        p = Path(r).expanduser()
        s = str(p.resolve()) if p.exists() else str(p)
        if s in seen:
            continue
        seen.add(s)
        if p.exists():
            out.append(p)
    return out


def discover_files(roots: List[Path]) -> List[Path]:
    found: List[Path] = []
    for root in roots:
        try:
            for p in root.rglob("*"):
                try:
                    if p.is_file() and is_allowed_table_file(p):
                        found.append(p)
                except Exception:
                    continue
        except Exception:
            continue
    return found


def score_path_for_role(path: Path, role: str) -> Tuple[int, List[str]]:
    s = str(path).lower().replace("\\", "/")
    name = path.name.lower()
    score = 0
    reasons: List[str] = []

    if "step9_v1/tables_supplementary" in s:
        score += 12; reasons.append("legacy_step9_supp")
    if "step_09" in s or "step9" in s:
        score += 6; reasons.append("step9_hint")
    if "step1" in s and role == "reference":
        score += 8; reasons.append("step1_reference_hint")

    patterns = {
        "final": ["table_s9_9_final_panel", "final_panel_with_all_step9_annotations", "final_panel", "final_12", "candidate"],
        "pairwise": ["table_s9_7_final_panel_pairwise_similarity", "pairwise_similarity", "similarity_matrix", "jaccard"],
        "reference": ["step1_retained_dataset", "retained_dataset", "reference", "training", "corpus"],
        "aa": ["table_s9_5_amino_acid", "amino_acid_enrichment", "aa_enrichment"],
        "kmer": ["table_s9_6_kmer", "kmer_enrichment", "3mer", "3_mer"],
        "paper": ["table_s9_11_paper_candidates", "paper_candidates", "harmonized"],
    }
    for patt in patterns.get(role, []):
        if patt in name:
            score += 10; reasons.append(f"name:{patt}")
    negative = ["readiness", "manifest", "requirements", "selection_audit", "stage_score", "score_shift", "figure", "source_data_all"]
    for n in negative:
        if n in name:
            score -= 8; reasons.append(f"penalty:{n}")
    return score, reasons


def resolve_input(role: str, explicit: Optional[str], roots: List[Path], log: List[Dict[str, str]]) -> Optional[Path]:
    if explicit:
        p = Path(explicit).expanduser()
        log.append({"role": role, "action": "explicit", "path": str(p), "reason": "user_supplied"})
        return p
    candidates = discover_files(roots)
    scored: List[Tuple[int, Path, List[str]]] = []
    for p in candidates:
        score, reasons = score_path_for_role(p, role)
        if score > 0:
            scored.append((score, p, reasons))
    if not scored:
        log.append({"role": role, "action": "not_found", "path": "", "reason": "no_safe_candidate"})
        return None
    scored.sort(key=lambda x: (-x[0], str(x[1])))
    score, p, reasons = scored[0]
    log.append({"role": role, "action": "auto", "path": str(p), "reason": f"score={score};" + ";".join(reasons)})
    return p


def read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suf == ".txt":
        for sep in [",", "\t", None]:
            try:
                return pd.read_csv(path, sep=sep) if sep else pd.read_csv(path, delim_whitespace=True)
            except Exception:
                pass
        raise ValueError(f"Could not parse text table: {path}")
    if suf in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suf in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table extension: {path}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def first_col(df: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    lower = {str(c).strip().lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in lower:
            return lower[a.lower()]
    return None


def standardize_final_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    out = df.copy()
    seq_col = first_col(out, ["sequence", "peptide", "seq", "aa_sequence", "generated_sequence"])
    if seq_col is None:
        raise RuntimeError("Final-panel table needs a sequence column.")
    out["sequence"] = out[seq_col].astype(str).map(clean_sequence)

    id_col = first_col(out, ["generated_id", "candidate_id", "peptide_id", "id", "seq_id", "name"])
    if id_col is None:
        out["generated_id"] = [f"C{i:02d}" for i in range(1, len(out) + 1)]
    else:
        out["generated_id"] = out[id_col].astype(str)

    alias_map = {
        "nearest_reference_similarity": [
            "nearest_reference_similarity", "nn_reference_similarity", "nearest_ref_similarity",
            "max_reference_similarity", "reference_nn_similarity", "nn_similarity_reference"
        ],
        "nearest_paper_candidate_similarity": [
            "nearest_paper_candidate_similarity", "nearest_candidate_similarity", "nearest_final_similarity",
            "paper_candidate_similarity", "candidate_context_similarity", "nn_paper_candidate_similarity"
        ],
    }
    for std, aliases in alias_map.items():
        if std not in out.columns:
            col = first_col(out, aliases)
            if col is not None:
                out[std] = pd.to_numeric(out[col], errors="coerce")
    return out


def clean_sequence(seq: str) -> str:
    if pd.isna(seq):
        return ""
    s = str(seq).strip().upper()
    return "".join(ch for ch in s if ch in AA_SET)


def kmers(seq: str, k: int = 3) -> set:
    seq = clean_sequence(seq)
    if len(seq) < k:
        return {seq} if seq else set()
    return {seq[i:i+k] for i in range(len(seq) - k + 1)}


def jaccard_sets(a: set, b: set) -> float:
    if not a and not b:
        return np.nan
    u = a | b
    if not u:
        return np.nan
    return len(a & b) / len(u)


def compute_pairwise_matrix(ids: Sequence[str], seqs: Sequence[str], k: int = 3) -> pd.DataFrame:
    ids = [str(x) for x in ids]
    ksets = [kmers(s, k=k) for s in seqs]
    n = len(ids)
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            mat[i, j] = 1.0 if i == j else jaccard_sets(ksets[i], ksets[j])
    return pd.DataFrame(mat, index=ids, columns=ids)


def read_pairwise_matrix(path: Path) -> pd.DataFrame:
    df = read_table(path)
    df = normalize_columns(df)
    # Matrix format: first column is index/candidate IDs and remaining are numeric columns
    first = df.columns[0]
    if not pd.api.types.is_numeric_dtype(df[first]):
        trial = df.set_index(first)
        numeric = trial.apply(pd.to_numeric, errors="coerce")
        # Accept if at least square-ish numeric matrix
        if numeric.shape[0] == numeric.shape[1] and numeric.notna().sum().sum() > 0:
            return numeric
    # Long format fallback
    cols = {c.lower(): c for c in df.columns}
    id1 = first_col(df, ["candidate_1", "id1", "source", "row", "candidate_i"])
    id2 = first_col(df, ["candidate_2", "id2", "target", "column", "candidate_j"])
    val = first_col(df, ["similarity", "jaccard", "jaccard_similarity", "3mer_jaccard", "value"])
    if id1 and id2 and val:
        tmp = df[[id1, id2, val]].copy()
        tmp[val] = pd.to_numeric(tmp[val], errors="coerce")
        mat = tmp.pivot_table(index=id1, columns=id2, values=val, aggfunc="first")
        # symmetrize where possible
        labels = sorted(set(mat.index.astype(str)) | set(mat.columns.astype(str)))
        mat = mat.reindex(index=labels, columns=labels)
        for a in labels:
            for b in labels:
                if pd.isna(mat.loc[a, b]) and not pd.isna(mat.loc[b, a]):
                    mat.loc[a, b] = mat.loc[b, a]
        np.fill_diagonal(mat.values, 1.0)
        return mat.apply(pd.to_numeric, errors="coerce")
    raise RuntimeError(f"Could not parse pairwise similarity matrix: {path}")


def max_similarity_to_reference(final_df: pd.DataFrame, ref_df: pd.DataFrame, k: int = 3, max_ref: int = 100000) -> pd.Series:
    ref = ref_df.copy()
    seq_col = first_col(ref, ["sequence", "peptide", "seq", "aa_sequence", "generated_sequence"])
    if seq_col is None:
        raise RuntimeError("Reference file needs a sequence column for fallback nearest-reference similarity.")
    ref_seqs = [clean_sequence(x) for x in ref[seq_col].astype(str).tolist()]
    ref_seqs = [s for s in ref_seqs if s]
    if len(ref_seqs) > max_ref:
        ref_seqs = ref_seqs[:max_ref]
    ref_sets = [kmers(s, k=k) for s in ref_seqs]
    vals = []
    for seq in final_df["sequence"].astype(str).tolist():
        sset = kmers(seq, k=k)
        if not sset or not ref_sets:
            vals.append(np.nan)
            continue
        mx = max(jaccard_sets(sset, rset) for rset in ref_sets)
        vals.append(float(mx))
    return pd.Series(vals, index=final_df.index)


def nearest_internal_similarity(pairwise: pd.DataFrame) -> pd.Series:
    mat = pairwise.copy().apply(pd.to_numeric, errors="coerce")
    vals = []
    for i, idx in enumerate(mat.index):
        arr = mat.iloc[i].to_numpy(dtype=float)
        if i < len(arr):
            arr[i] = np.nan
        vals.append(float(np.nanmax(arr)) if np.isfinite(arr).any() else np.nan)
    return pd.Series(vals, index=mat.index)


def percentile_summary(vals: np.ndarray) -> Dict[str, float]:
    vals = np.asarray([v for v in vals if pd.notna(v)], dtype=float)
    if len(vals) == 0:
        return {"n": 0, "median": np.nan, "p90": np.nan, "max": np.nan, "q1": np.nan, "q3": np.nan}
    return {
        "n": int(len(vals)),
        "median": float(np.nanmedian(vals)),
        "p90": float(np.nanpercentile(vals, 90)),
        "max": float(np.nanmax(vals)),
        "q1": float(np.nanpercentile(vals, 25)),
        "q3": float(np.nanpercentile(vals, 75)),
    }


def aa_enrichment(final_seqs: Sequence[str], ref_seqs: Sequence[str], pseudocount: float = 1e-6) -> pd.DataFrame:
    fcnt = Counter(ch for s in final_seqs for ch in clean_sequence(s) if ch in AA_SET)
    rcnt = Counter(ch for s in ref_seqs for ch in clean_sequence(s) if ch in AA_SET)
    f_total = sum(fcnt.values())
    r_total = sum(rcnt.values())
    rows = []
    for aa in AA_ORDER:
        f = fcnt.get(aa, 0)
        r = rcnt.get(aa, 0)
        f_freq = (f + pseudocount) / (f_total + pseudocount * len(AA_ORDER)) if f_total else np.nan
        r_freq = (r + pseudocount) / (r_total + pseudocount * len(AA_ORDER)) if r_total else np.nan
        log2e = float(np.log2(f_freq / r_freq)) if pd.notna(f_freq) and pd.notna(r_freq) and r_freq > 0 else np.nan
        rows.append({"amino_acid": aa, "final_count": f, "reference_count": r, "final_frequency": f_freq, "reference_frequency": r_freq, "log2_enrichment_vs_reference": log2e})
    return pd.DataFrame(rows)


def kmer_enrichment(final_seqs: Sequence[str], ref_seqs: Sequence[str], k: int = 3, pseudocount: float = 1e-6, top_n: int = 25) -> pd.DataFrame:
    fcnt = Counter(km for s in final_seqs for km in kmers(s, k=k))
    rcnt = Counter(km for s in ref_seqs for km in kmers(s, k=k))
    universe = sorted(set(fcnt) | set(rcnt))
    f_total = sum(fcnt.values())
    r_total = sum(rcnt.values())
    rows = []
    denom = max(len(universe), 1)
    for km in universe:
        f = fcnt.get(km, 0)
        r = rcnt.get(km, 0)
        f_freq = (f + pseudocount) / (f_total + pseudocount * denom) if f_total else np.nan
        r_freq = (r + pseudocount) / (r_total + pseudocount * denom) if r_total else np.nan
        log2e = float(np.log2(f_freq / r_freq)) if pd.notna(f_freq) and pd.notna(r_freq) and r_freq > 0 else np.nan
        rows.append({"kmer": km, "final_count": f, "reference_count": r, "final_frequency": f_freq, "reference_frequency": r_freq, "log2_enrichment_vs_reference": log2e})
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    out = out.sort_values(["log2_enrichment_vs_reference", "final_count"], ascending=[False, False]).head(top_n)
    return out


def frac_of_categories(seqs: Sequence[str]) -> Dict[str, float]:
    seqs = [clean_sequence(s) for s in seqs]
    total = sum(len(s) for s in seqs)
    if total == 0:
        return {"Hydrophobic": np.nan, "Aromatic": np.nan, "Positive": np.nan, "Negative": np.nan, "Polar": np.nan}
    def frac(res_set: set) -> float:
        return sum(sum(ch in res_set for ch in s) for s in seqs) / total
    return {
        "Hydrophobic": frac(AA_HYDROPHOBIC),
        "Aromatic": frac(AA_AROMATIC),
        "Positive": frac(AA_POSITIVE),
        "Negative": frac(AA_NEGATIVE),
        "Polar": frac(AA_POLAR),
    }


def get_ref_sequences(ref_df: pd.DataFrame) -> List[str]:
    seq_col = first_col(ref_df, ["sequence", "peptide", "seq", "aa_sequence", "generated_sequence"])
    if seq_col is None:
        return []
    return [clean_sequence(x) for x in ref_df[seq_col].astype(str).tolist() if clean_sequence(x)]


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": PLOS["white"],
        "axes.facecolor": PLOS["white"],
        "savefig.facecolor": PLOS["white"],
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 15,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": PLOS["grid"],
        "grid.linewidth": 0.8,
        "grid.alpha": 0.85,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_axis(ax, grid_axis: str = "y") -> None:
    ax.spines["left"].set_color(PLOS["edge"])
    ax.spines["bottom"].set_color(PLOS["edge"])
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(colors=PLOS["dark"], width=0.8, length=4)
    ax.set_axisbelow(True)
    ax.grid(True, axis=grid_axis, color=PLOS["grid"], linewidth=0.8, zorder=0)
    if grid_axis == "y":
        ax.grid(False, axis="x")
    elif grid_axis == "x":
        ax.grid(False, axis="y")
    elif grid_axis == "both":
        ax.grid(True, axis="both", color=PLOS["grid"], linewidth=0.8, zorder=0)


def add_panel_label(ax, label: str, x: float = -0.14, y: float = 1.08, fontsize: int = 18) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize, fontweight="bold", ha="left", va="top", color=PLOS["dark"])


def draw_violin(ax, data: Sequence[float], pos: float, color: str, width: float = 0.72) -> None:
    arr = np.asarray([x for x in data if pd.notna(x)], dtype=float)
    if len(arr) == 0:
        return
    parts = ax.violinplot([arr], positions=[pos], widths=width, showmeans=False, showmedians=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.78)
    q1, med, q3 = np.nanpercentile(arr, [25, 50, 75])
    ax.plot([pos, pos], [q1, q3], color=PLOS["dark"], linewidth=1.2, zorder=4)
    ax.scatter([pos], [med], s=34, facecolor=PLOS["white"], edgecolor=PLOS["dark"], linewidth=0.8, zorder=5)
    # Subtle points
    rng = np.random.default_rng(42 + int(pos * 10))
    jitter = rng.uniform(-0.065, 0.065, size=len(arr))
    ax.scatter(np.full(len(arr), pos) + jitter, arr, s=10, facecolor=PLOS["white"], edgecolor=PLOS["dark"], linewidth=0.35, alpha=0.65, zorder=4)


def save_multi(fig: plt.Figure, out_base: Path, dpi: int) -> Dict[str, str]:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    for ext in [".png", ".pdf", ".tiff"]:
        p = out_base.with_suffix(ext)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        paths[ext.lstrip(".")] = str(p)
    plt.close(fig)
    return paths


def short_label(x: str, max_len: int = 12) -> str:
    s = str(x)
    return s if len(s) <= max_len else s[:max_len]


def sorted_pairwise_matrix(pairwise: pd.DataFrame, final_ids: Sequence[str]) -> pd.DataFrame:
    mat = pairwise.copy()
    mat.index = mat.index.astype(str)
    mat.columns = mat.columns.astype(str)
    ids = [str(i) for i in final_ids]
    available = [i for i in ids if i in mat.index and i in mat.columns]
    if len(available) >= 2:
        mat = mat.loc[available, available]
    else:
        # keep original square matrix
        pass
    numeric = mat.apply(pd.to_numeric, errors="coerce")
    # order by mean off-diagonal similarity to make layout intentional
    vals = numeric.to_numpy(dtype=float)
    np.fill_diagonal(vals, np.nan)
    avg = np.nanmean(vals, axis=1)
    order = np.argsort(-np.nan_to_num(avg, nan=-1))
    numeric = numeric.iloc[order, :].iloc[:, order]
    return numeric


def residue_category_df(final_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    """Return residue-category fractions for reference corpus and final panel."""
    final_seqs = final_df["sequence"].astype(str).tolist()
    ref_seqs = get_ref_sequences(reference_df)
    ref_cat = frac_of_categories(ref_seqs)
    final_cat = frac_of_categories(final_seqs)
    categories = ["Hydrophobic", "Aromatic", "Positive", "Negative", "Polar"]
    return pd.DataFrame({
        "category": categories,
        "reference_fraction": [ref_cat[c] for c in categories],
        "final_panel_fraction": [final_cat[c] for c in categories],
    })


def write_pairwise_source_and_heatmap(
    ax,
    pairwise: pd.DataFrame,
    final_ids: Sequence[str],
    title: str,
    colorbar_label: str = "3-mer Jaccard similarity",
    panel_label: Optional[str] = None,
) -> pd.DataFrame:
    """Draw polished supplementary heatmap and return ordered matrix."""
    mat = sorted_pairwise_matrix(pairwise, final_ids)
    arr = mat.to_numpy(dtype=float)
    if arr.shape[0] == 0:
        raise RuntimeError("Pairwise matrix has no rows after ordering.")
    diag_mask = np.eye(arr.shape[0], dtype=bool)
    offdiag = arr.copy()
    offdiag[diag_mask] = np.nan
    max_offdiag = float(np.nanmax(offdiag)) if np.isfinite(offdiag).any() else 0.20
    vmax = min(1.0, max(0.20, math.ceil(max_offdiag * 10) / 10))
    cmap = LinearSegmentedColormap.from_list(
        "oncopep_similarity",
        [PLOS["white"], PLOS["mint"], PLOS["blue"]],
    )
    plot_arr = arr.copy()
    plot_arr[diag_mask] = np.nan
    masked = np.ma.masked_invalid(plot_arr)
    cmap.set_bad(color=PLOS["pale"])
    im = ax.imshow(masked, vmin=0, vmax=vmax, cmap=cmap, aspect="equal")
    labels = [short_label(x, 12) for x in mat.index.tolist()]
    ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=8)
    ax.set_title(title, loc="left", pad=8)
    if panel_label:
        add_panel_label(ax, panel_label, x=-0.08, y=1.06, fontsize=16)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color=PLOS["white"], linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.042, pad=0.035)
    cbar.set_label(colorbar_label)
    return mat


def plot_figure5(
    final_df: pd.DataFrame,
    pairwise: pd.DataFrame,
    reference_df: pd.DataFrame,
    source_data_dir: Path,
    out_base: Path,
    dpi: int,
) -> Dict[str, str]:
    """Generate final Figure 5 after approved panel swap.

    Final structure:
      A. NN similarity context
      B. Similarity summary
      C. Residue-category context
    """
    set_style()

    ref_vals = pd.to_numeric(final_df["nearest_reference_similarity"], errors="coerce").dropna().to_numpy(dtype=float)
    cand_col = "nearest_paper_candidate_similarity"
    cand_vals = pd.to_numeric(final_df[cand_col], errors="coerce").dropna().to_numpy(dtype=float)

    summary_rows = []
    for name, arr in [("Reference context", ref_vals), ("Final-panel context", cand_vals)]:
        st = percentile_summary(arr)
        summary_rows.append({"context": name, **st})
    summary_df = pd.DataFrame(summary_rows)

    panel_a_df = pd.concat([
        pd.DataFrame({"context": "Reference context", "nearest_neighbor_similarity": ref_vals}),
        pd.DataFrame({"context": "Final-panel context", "nearest_neighbor_similarity": cand_vals}),
    ], ignore_index=True)
    panel_a_df.to_csv(source_data_dir / "Figure_5_panel_a_source_data.csv", index=False)

    panel_b_rows = []
    for _, r in summary_df.iterrows():
        for metric in ["median", "p90", "max"]:
            panel_b_rows.append({"context": r["context"], "summary_metric": metric, "similarity": r[metric], "n": r["n"]})
    panel_b_df = pd.DataFrame(panel_b_rows)
    panel_b_df.to_csv(source_data_dir / "Figure_5_panel_b_source_data.csv", index=False)

    cat_df = residue_category_df(final_df, reference_df)
    cat_df.to_csv(source_data_dir / "Figure_5_panel_c_source_data.csv", index=False)

    combined = pd.concat([
        panel_a_df.assign(panel="Figure_5A"),
        panel_b_df.assign(panel="Figure_5B"),
        cat_df.assign(panel="Figure_5C"),
    ], ignore_index=True, sort=False)
    combined.to_csv(source_data_dir / "Figure_5_source_data_all_panels.csv", index=False)

    fig = plt.figure(figsize=(15.8, 5.15))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[0.88, 0.92, 1.10], wspace=0.50)

    # A violin
    ax = fig.add_subplot(gs[0, 0])
    add_panel_label(ax, "A", x=-0.18, y=1.08)
    style_axis(ax, grid_axis="y")
    draw_violin(ax, ref_vals, 0, PLOS["blue"], width=0.78)
    draw_violin(ax, cand_vals, 1, PLOS["coral"], width=0.78)
    ymax = max(0.16, float(np.nanmax(np.r_[ref_vals, cand_vals])) * 1.22 if len(np.r_[ref_vals, cand_vals]) else 0.16)
    ax.set_ylim(0, ymax)
    ax.set_xticks([0, 1], ["Reference\ncontext", "Final-panel\ncontext"])
    ax.set_ylabel("Nearest-neighbor similarity")
    ax.set_title("NN similarity context", pad=10)

    # B summary bars
    ax = fig.add_subplot(gs[0, 1])
    add_panel_label(ax, "B", x=-0.18, y=1.08)
    style_axis(ax, grid_axis="y")
    metrics = ["median", "p90", "max"]
    metric_labels = ["Median", "P90", "Max"]
    x = np.arange(len(metrics), dtype=float)
    width = 0.32
    ref_s = summary_df.set_index("context").loc["Reference context"]
    cand_s = summary_df.set_index("context").loc["Final-panel context"]
    ref_bar = [float(ref_s[m]) for m in metrics]
    cand_bar = [float(cand_s[m]) for m in metrics]
    ax.bar(x - width/2, ref_bar, width=width, color=PLOS["blue"], edgecolor=PLOS["dark"], linewidth=0.5, label="Reference context", zorder=3)
    ax.bar(x + width/2, cand_bar, width=width, color=PLOS["coral"], edgecolor=PLOS["dark"], linewidth=0.5, label="Final-panel context", zorder=3)
    ax.set_xticks(x, metric_labels)
    ax.set_ylabel("Nearest-neighbor similarity")
    ymax_b = max(0.16, float(np.nanmax(np.r_[ref_bar, cand_bar])) * 1.22)
    ax.set_ylim(0, ymax_b)
    ax.set_title("Similarity summary", pad=10)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True, handlelength=1.0, columnspacing=0.85, handletextpad=0.45, borderaxespad=0.3)
    leg.get_frame().set_edgecolor(PLOS["edge"])
    leg.get_frame().set_linewidth(0.8)

    # C residue-category context
    ax = fig.add_subplot(gs[0, 2])
    add_panel_label(ax, "C", x=-0.14, y=1.08)
    style_axis(ax, grid_axis="y")
    categories = cat_df["category"].astype(str).tolist()
    x = np.arange(len(categories), dtype=float)
    width = 0.32
    ref_cat_vals = cat_df["reference_fraction"].to_numpy(dtype=float)
    final_cat_vals = cat_df["final_panel_fraction"].to_numpy(dtype=float)
    ax.bar(x - width/2, ref_cat_vals, width=width, color=PLOS["light"], edgecolor=PLOS["dark"], linewidth=0.4, label="Reference ACP corpus", zorder=3)
    ax.bar(x + width/2, final_cat_vals, width=width, color=PLOS["blue"], edgecolor=PLOS["dark"], linewidth=0.4, label="Final panel", zorder=3)
    ax.set_xticks(x, categories, rotation=18, ha="right")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, max(0.34, float(np.nanmax(np.r_[ref_cat_vals, final_cat_vals])) * 1.18))
    ax.set_title("Residue-category context", pad=10)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=True, handlelength=1.0, columnspacing=0.8)
    leg.get_frame().set_edgecolor(PLOS["edge"])
    leg.get_frame().set_linewidth(0.8)

    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.22, top=0.90)
    return save_multi(fig, out_base, dpi=dpi)


def plot_supplementary_s14(
    final_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    pairwise: pd.DataFrame,
    aa_df: pd.DataFrame,
    kmer_df: pd.DataFrame,
    source_data_dir: Path,
    out_base: Path,
    dpi: int,
) -> Dict[str, str]:
    """Generate polished two-panel Supplementary Figure S14.

    Final plotted structure:
      A. Amino-acid enrichment
      B. Top enriched 3-mers

    Pairwise final-panel similarity is not plotted here. It is retained as
    source data and documented in step9_removed_panel_report.csv to avoid a
    crowded supplementary figure.
    """
    set_style()

    final_seqs = final_df["sequence"].astype(str).tolist()
    ref_seqs = get_ref_sequences(reference_df)

    # Normalize AA enrichment table or compute fallback
    aa = normalize_columns(aa_df.copy()) if aa_df is not None else pd.DataFrame()
    if "log2_enrichment_vs_reference" not in aa.columns:
        col = first_col(aa, ["log2_enrichment", "enrichment", "log2fc", "log2_fold_change"])
        if col:
            aa["log2_enrichment_vs_reference"] = pd.to_numeric(aa[col], errors="coerce")
    if "amino_acid" not in aa.columns:
        col = first_col(aa, ["aa", "residue", "amino acid"])
        if col:
            aa["amino_acid"] = aa[col].astype(str)
    if "amino_acid" not in aa.columns or "log2_enrichment_vs_reference" not in aa.columns:
        aa = aa_enrichment(final_seqs, ref_seqs)
    aa["amino_acid"] = aa["amino_acid"].astype(str)
    aa["log2_enrichment_vs_reference"] = pd.to_numeric(aa["log2_enrichment_vs_reference"], errors="coerce")

    # Normalize k-mer enrichment table or compute fallback
    km = normalize_columns(kmer_df.copy()) if kmer_df is not None else pd.DataFrame()
    if "log2_enrichment_vs_reference" not in km.columns:
        col = first_col(km, ["log2_enrichment", "enrichment", "log2fc", "log2_fold_change"])
        if col:
            km["log2_enrichment_vs_reference"] = pd.to_numeric(km[col], errors="coerce")
    if "kmer" not in km.columns:
        col = first_col(km, ["3mer", "k_mer", "motif", "token"])
        if col:
            km["kmer"] = km[col].astype(str)
    if "kmer" not in km.columns or "log2_enrichment_vs_reference" not in km.columns:
        km = kmer_enrichment(final_seqs, ref_seqs, k=3, top_n=25)
    km["kmer"] = km["kmer"].astype(str)
    km["log2_enrichment_vs_reference"] = pd.to_numeric(km["log2_enrichment_vs_reference"], errors="coerce")

    # Retain pairwise matrix as source data/report, but do not plot it.
    final_ids = final_df["generated_id"].astype(str).tolist()
    mat = sorted_pairwise_matrix(pairwise, final_ids)

    aa_out = source_data_dir / "Supplementary_Figure_S14_panel_a_source_data.csv"
    km_out = source_data_dir / "Supplementary_Figure_S14_panel_b_source_data.csv"
    matrix_out = source_data_dir / "step9_pairwise_similarity_matrix.csv"
    removed_out = source_data_dir / "step9_removed_panel_report.csv"

    aa.to_csv(aa_out, index=False)
    km.to_csv(km_out, index=False)
    mat.reset_index().rename(columns={"index": "candidate_id"}).to_csv(matrix_out, index=False)

    removed_report = pd.DataFrame([
        {
            "removed_panel": "Supplementary_Figure_S14C_pairwise_similarity_heatmap",
            "decision": "removed_from_plotted_supplementary_figure",
            "reason": "Dense candidate-level heatmap was visually weaker than the two-panel composition figure and is retained as source data/report instead.",
            "retained_source_data": str(matrix_out),
            "manuscript_role": "source_data_internal_diversity_support",
        }
    ])
    removed_report.to_csv(removed_out, index=False)

    pd.concat([
        aa.assign(panel="S14A"),
        km.assign(panel="S14B"),
    ], ignore_index=True, sort=False).to_csv(source_data_dir / "Supplementary_Figure_S14_source_data_all_panels.csv", index=False)

    # Two-panel layout: large AA enrichment + k-mer enrichment
    fig = plt.figure(figsize=(12.6, 5.65))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.08, 1.0], wspace=0.36)

    # A amino-acid enrichment
    ax = fig.add_subplot(gs[0, 0])
    add_panel_label(ax, "A", x=-0.07, y=1.04, fontsize=17)
    style_axis(ax, grid_axis="x")
    tmp = aa[["amino_acid", "log2_enrichment_vs_reference"]].dropna().copy()
    tmp = tmp.sort_values("log2_enrichment_vs_reference", ascending=True)
    vals = tmp["log2_enrichment_vs_reference"].to_numpy(dtype=float)
    colors = [PLOS["light"] if v < 0 else PLOS["brown"] for v in vals]
    ax.barh(tmp["amino_acid"].astype(str), vals, color=colors, edgecolor=PLOS["dark"], linewidth=0.35, zorder=3)
    ax.axvline(0, color=PLOS["dark"], linewidth=0.9, zorder=4)
    ax.set_xlabel("log2 enrichment vs reference")
    ax.set_title("Amino-acid enrichment", loc="center", pad=10)

    # B top enriched 3-mers with unique restrained palette
    ax = fig.add_subplot(gs[0, 1])
    add_panel_label(ax, "B", x=-0.08, y=1.04, fontsize=17)
    style_axis(ax, grid_axis="x")
    tmp = km[["kmer", "log2_enrichment_vs_reference"]].dropna().copy()
    tmp = tmp.sort_values("log2_enrichment_vs_reference", ascending=False).head(8)
    tmp = tmp.sort_values("log2_enrichment_vs_reference", ascending=True)
    vals = tmp["log2_enrichment_vs_reference"].to_numpy(dtype=float)
    # Each plotted 3-mer receives a unique, restrained color.  The palette is
    # deliberately limited to OncoPep/PLOS-compatible tones and muted variants,
    # avoiding decorative rainbow coloring while preventing repeated bars.
    unique_kmer_palette = [
        PLOS["mint"],      # muted green
        "#7FB8C8",         # soft blue-green
        PLOS["blue"],      # OncoPep blue
        "#4F8FA3",         # deeper blue
        PLOS["brown"],     # warm brown
        "#C99A66",         # light brown
        PLOS["coral"],     # coral
        PLOS["charcoal"],  # muted charcoal
        "#9FB7A8",         # desaturated mint-gray fallback
        "#A87E73",         # muted clay fallback
        "#6FA7B3",         # extra teal fallback
        "#8B7C80",         # extra charcoal fallback
    ]
    colors = unique_kmer_palette[:len(vals)]
    ax.barh(tmp["kmer"].astype(str), vals, color=colors, edgecolor=PLOS["dark"], linewidth=0.35, zorder=3)
    ax.set_xlabel("log2 enrichment vs reference")
    ax.set_title("Top enriched 3-mers", loc="center", pad=10)

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.15, top=0.90)
    return save_multi(fig, out_base, dpi=dpi)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate OncoPep Step 9 Figure 5 and Supplementary Figure S14.", allow_abbrev=False)
    p.add_argument("--step9-root", default=DEFAULT_STEP9_ROOT)
    p.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    p.add_argument("--legacy-step9-root", default=DEFAULT_LEGACY_STEP9_ROOT)
    p.add_argument("--legacy-reference-file", default=DEFAULT_LEGACY_REFERENCE)
    p.add_argument("--final-panel-file", default=None)
    p.add_argument("--reference-file", default=None)
    p.add_argument("--pairwise-file", default=None)
    p.add_argument("--aa-enrichment-file", default=None)
    p.add_argument("--kmer-enrichment-file", default=None)
    p.add_argument("--paper-candidates-file", default=None)
    p.add_argument("--similarity-k", type=int, default=3)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--no-supplementary", action="store_true")
    p.add_argument("--max-reference-fallback", type=int, default=100000)
    p.add_argument("--quiet", action="store_true")
    return p


def load_inputs(args, checks: List[CheckResult], log: List[Dict[str, str]]) -> Dict[str, pd.DataFrame]:
    roots = safe_roots(args.project_root, args.legacy_step9_root, Path(args.project_root).parent if args.project_root else None)
    data: Dict[str, pd.DataFrame] = {}
    paths = {
        "final": resolve_input("final", args.final_panel_file, roots, log),
        "pairwise": resolve_input("pairwise", args.pairwise_file, roots, log),
        "reference": Path(args.reference_file) if args.reference_file else (Path(args.legacy_reference_file) if args.legacy_reference_file and Path(args.legacy_reference_file).exists() else resolve_input("reference", None, roots, log)),
        "aa": resolve_input("aa", args.aa_enrichment_file, roots, log),
        "kmer": resolve_input("kmer", args.kmer_enrichment_file, roots, log),
        "paper": resolve_input("paper", args.paper_candidates_file, roots, log),
    }
    data["_paths"] = pd.DataFrame([{"role": k, "path": str(v) if v else ""} for k, v in paths.items()])

    # Required final file
    if paths["final"] is None or not paths["final"].exists():
        checks.append(CheckResult("final_panel_file", "FAIL", "Final-panel annotation table missing. Supply --final-panel-file."))
        return data
    try:
        data["final"] = standardize_final_df(read_table(paths["final"]))
        checks.append(CheckResult("final_panel_file", "PASS", f"Loaded final panel: {paths['final']} ({len(data['final'])} rows)."))
    except Exception as e:
        checks.append(CheckResult("final_panel_file", "FAIL", f"Could not read final panel {paths['final']}: {e}"))
        return data

    # Reference is required for context fallback and S14
    if paths["reference"] is not None and paths["reference"].exists():
        try:
            data["reference"] = read_table(paths["reference"])
            checks.append(CheckResult("reference_file", "PASS", f"Loaded reference corpus: {paths['reference']} ({len(data['reference'])} rows)."))
        except Exception as e:
            checks.append(CheckResult("reference_file", "WARN", f"Could not read reference file {paths['reference']}: {e}"))
    else:
        checks.append(CheckResult("reference_file", "WARN", "Reference corpus file not found; nearest-reference fallback and S14 may be unavailable."))

    if paths["pairwise"] is not None and paths["pairwise"].exists():
        try:
            data["pairwise"] = read_pairwise_matrix(paths["pairwise"])
            checks.append(CheckResult("pairwise_file", "PASS", f"Loaded pairwise similarity matrix: {paths['pairwise']}."))
        except Exception as e:
            checks.append(CheckResult("pairwise_file", "WARN", f"Could not parse pairwise file; will compute from sequences if possible: {e}"))
    else:
        checks.append(CheckResult("pairwise_file", "WARN", "Pairwise similarity matrix not found; computing 3-mer Jaccard matrix from final-panel sequences."))

    for role, key in [("aa", "aa"), ("kmer", "kmer")]:
        if paths[role] is not None and paths[role].exists():
            try:
                data[key] = read_table(paths[role])
                checks.append(CheckResult(f"{role}_enrichment_file", "PASS", f"Loaded {role} enrichment table: {paths[role]}."))
            except Exception as e:
                checks.append(CheckResult(f"{role}_enrichment_file", "WARN", f"Could not read {role} enrichment table; will compute if possible: {e}"))
        else:
            checks.append(CheckResult(f"{role}_enrichment_file", "WARN", f"{role} enrichment table not found; will compute from final/reference sequences if possible."))

    return data


def prepare_data(data: Dict[str, pd.DataFrame], args, checks: List[CheckResult]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    final_df = data["final"].copy()
    reference_df = data.get("reference")

    if "pairwise" in data:
        pairwise = data["pairwise"].copy()
    else:
        pairwise = compute_pairwise_matrix(final_df["generated_id"].tolist(), final_df["sequence"].tolist(), k=args.similarity_k)
        checks.append(CheckResult("pairwise_computed", "WARN", f"Computed pairwise {args.similarity_k}-mer Jaccard matrix from final-panel sequences."))

    # Fill nearest reference if missing
    if "nearest_reference_similarity" not in final_df.columns or pd.to_numeric(final_df.get("nearest_reference_similarity", pd.Series(dtype=float)), errors="coerce").notna().sum() == 0:
        if reference_df is not None:
            final_df["nearest_reference_similarity"] = max_similarity_to_reference(final_df, reference_df, k=args.similarity_k, max_ref=args.max_reference_fallback)
            checks.append(CheckResult("nearest_reference_similarity", "WARN", "Computed nearest-reference similarity from sequences because input column was unavailable."))
        else:
            checks.append(CheckResult("nearest_reference_similarity", "FAIL", "Nearest-reference similarity is missing and no reference file is available."))
    else:
        final_df["nearest_reference_similarity"] = pd.to_numeric(final_df["nearest_reference_similarity"], errors="coerce")
        checks.append(CheckResult("nearest_reference_similarity", "PASS", "Using nearest-reference similarity column from final-panel table."))

    if "nearest_paper_candidate_similarity" not in final_df.columns or pd.to_numeric(final_df.get("nearest_paper_candidate_similarity", pd.Series(dtype=float)), errors="coerce").notna().sum() == 0:
        internal = nearest_internal_similarity(pairwise)
        # map by IDs
        id_map = {str(k): v for k, v in internal.items()}
        final_df["nearest_paper_candidate_similarity"] = final_df["generated_id"].astype(str).map(id_map)
        if final_df["nearest_paper_candidate_similarity"].isna().all():
            final_df["nearest_paper_candidate_similarity"] = internal.to_numpy()[:len(final_df)]
        checks.append(CheckResult("nearest_candidate_similarity", "WARN", "Computed candidate-context similarity from the pairwise final-panel matrix because input column was unavailable."))
    else:
        final_df["nearest_paper_candidate_similarity"] = pd.to_numeric(final_df["nearest_paper_candidate_similarity"], errors="coerce")
        checks.append(CheckResult("nearest_candidate_similarity", "PASS", "Using candidate-context similarity column from final-panel table."))

    aa_df = data.get("aa")
    km_df = data.get("kmer")
    if reference_df is not None:
        final_seqs = final_df["sequence"].astype(str).tolist()
        ref_seqs = get_ref_sequences(reference_df)
        if aa_df is None:
            aa_df = aa_enrichment(final_seqs, ref_seqs)
            checks.append(CheckResult("aa_enrichment", "WARN", "Computed amino-acid enrichment from final/reference sequences."))
        if km_df is None:
            km_df = kmer_enrichment(final_seqs, ref_seqs, k=args.similarity_k, top_n=25)
            checks.append(CheckResult("kmer_enrichment", "WARN", "Computed k-mer enrichment from final/reference sequences."))

    return final_df, pairwise, reference_df, aa_df, km_df


def write_reports(dirs: OutputDirs, checks: List[CheckResult], discovery: List[Dict[str, str]], files: Dict[str, str], data_paths: pd.DataFrame) -> Dict[str, str]:
    n_fail = sum(c.status == "FAIL" for c in checks)
    n_warn = sum(c.status == "WARN" for c in checks)
    status = "FAIL" if n_fail else ("WARN" if n_warn else "PASS")
    score = max(0, 100 - 15 * n_fail - 2 * n_warn)

    report = dirs.reports / "step9_readiness_report.md"
    lines = [
        "# OncoPep Step 9 readiness report\n",
        f"**Script version:** `{SCRIPT_VERSION}`\n",
        f"**Overall status:** `{status}`\n",
        f"**Estimated readiness score:** `{score}/100`\n",
        "## Validation checks\n",
    ]
    for c in checks:
        lines.append(f"- **[{c.status}] {c.name}**: {c.message}\n")
    lines.append("\n## Input paths\n")
    try:
        lines.append(data_paths.to_markdown(index=False) + "\n")
    except Exception:
        lines.append(data_paths.to_csv(index=False) + "\n")
    lines.append("\n## Discovery log\n")
    for rec in discovery:
        lines.append(f"- role=`{rec.get('role','')}` action=`{rec.get('action','')}` path=`{rec.get('path','')}` reason=`{rec.get('reason','')}`\n")
    lines.append("\n## Output files\n")
    for k, v in files.items():
        lines.append(f"- **{k}**: `{v}`\n")
    lines.append("\n## Non-duplication statement\n")
    lines.append("This Step 9 package uses contextual nearest-neighbor similarity, summary similarity statistics, main-figure residue-category context, and a two-panel supplementary sequence-composition figure. The pairwise final-panel similarity heatmap was removed from the plotted supplement and retained as source data/report to avoid a crowded supplementary figure. The package excludes selection-audit and prioritization-score-shift panels because those belong to Step 8/Figure 4.\n")
    report.write_text("\n".join(lines), encoding="utf-8")
    files["readiness_report"] = str(report)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "output_root": str(dirs.root),
        "checks": [asdict(c) for c in checks],
        "files": files,
        "input_paths": data_paths.to_dict(orient="records"),
        "discovery_log": discovery,
    }
    manifest_path = dirs.reports / "step9_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    files["manifest_json"] = str(manifest_path)

    readme = dirs.reports / "README_step9_outputs.txt"
    readme.write_text(
        f"OncoPep Step 9 output package\nScript version: {SCRIPT_VERSION}\n\n"
        "Main Figure 5: contextual similarity support and internal diversity of the final OncoPep panel.\n"
        "Supplementary Figure S14: compositional context of the final OncoPep panel.\n\n"
        "This package intentionally excludes Step 8 selection-audit/prioritization score-shift panels.\n",
        encoding="utf-8",
    )
    files["readme"] = str(readme)

    req = dirs.reports / "requirements_step9_minimal.txt"
    req.write_text("python>=3.10\nnumpy>=1.23\npandas>=1.5\nmatplotlib>=3.6\nopenpyxl>=3.0\n", encoding="utf-8")
    files["requirements_file"] = str(req)

    try:
        if "__file__" in globals():
            src = Path(__file__).resolve()
            if src.exists():
                dst = dirs.code / "OncoPep_step9_PLOS_contextual_similarity_diversity_final.py"
                shutil.copy2(src, dst)
                files["code_snapshot"] = str(dst)
    except Exception:
        pass
    return files


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args, unknown = parser.parse_known_args(clean_argv(argv))
    if not args.quiet:
        print(f"Hard-fix version: {SCRIPT_VERSION}")
        if unknown:
            print(f"Ignored unknown arguments: {unknown}")

    dirs = ensure_dirs(Path(args.step9_root).expanduser().resolve())
    checks: List[CheckResult] = []
    discovery: List[Dict[str, str]] = []
    data = load_inputs(args, checks, discovery)
    files: Dict[str, str] = {}

    if any(c.status == "FAIL" for c in checks):
        data_paths = data.get("_paths", pd.DataFrame())
        write_reports(dirs, checks, discovery, files, data_paths)
        msg = "Required Step 9 inputs are missing. Review step9_readiness_report.md."
        if is_notebook():
            raise RuntimeError(msg)
        print("ERROR:", msg, file=sys.stderr)
        return 2

    final_df, pairwise, reference_df, aa_df, km_df = prepare_data(data, args, checks)

    # Check post-preparation fatal conditions
    if final_df["nearest_reference_similarity"].isna().all():
        checks.append(CheckResult("plot_readiness", "FAIL", "Figure 5A cannot be plotted: nearest-reference similarity unavailable."))
    if final_df["nearest_paper_candidate_similarity"].isna().all():
        checks.append(CheckResult("plot_readiness", "FAIL", "Figure 5A/B cannot be plotted: candidate-context similarity unavailable."))
    if pairwise.shape[0] < 2:
        checks.append(CheckResult("plot_readiness", "FAIL", "Pairwise internal-diversity source data cannot be summarized: pairwise matrix has fewer than two candidates."))
    if reference_df is None:
        checks.append(CheckResult("plot_readiness", "FAIL", "Figure 5C cannot be plotted: reference corpus is required for residue-category context."))

    # Save prepared final panel and matrix
    final_df.to_csv(dirs.source_data / "step9_final_panel_similarity_context_table.csv", index=False)
    pairwise.reset_index().rename(columns={"index": "candidate_id"}).to_csv(dirs.source_data / "step9_final_panel_similarity_matrix.csv", index=False)
    files["step9_final_panel_similarity_context_table"] = str(dirs.source_data / "step9_final_panel_similarity_context_table.csv")
    files["step9_final_panel_similarity_matrix"] = str(dirs.source_data / "step9_final_panel_similarity_matrix.csv")

    if any(c.status == "FAIL" for c in checks):
        write_reports(dirs, checks, discovery, files, data.get("_paths", pd.DataFrame()))
        msg = "Step 9 plotting readiness failed. Review step9_readiness_report.md."
        if is_notebook():
            raise RuntimeError(msg)
        print("ERROR:", msg, file=sys.stderr)
        return 2

    fig5_files = plot_figure5(
        final_df=final_df,
        pairwise=pairwise,
        reference_df=reference_df,
        source_data_dir=dirs.source_data,
        out_base=dirs.main_figure / "Figure_5_contextual_similarity_diversity",
        dpi=args.dpi,
    )
    for k, v in fig5_files.items():
        files[f"Figure_5_{k}"] = v
    checks.append(CheckResult("Figure_5", "PASS", "Generated redesigned Figure 5 with contextual NN similarity, NN summary, and residue-category context."))

    if not args.no_supplementary:
        if reference_df is not None and aa_df is not None and km_df is not None:
            s14_files = plot_supplementary_s14(
                final_df=final_df,
                reference_df=reference_df,
                pairwise=pairwise,
                aa_df=aa_df,
                kmer_df=km_df,
                source_data_dir=dirs.source_data,
                out_base=dirs.supplementary_figures / "Supplementary_Figure_S14_additional_context",
                dpi=args.dpi,
            )
            for k, v in s14_files.items():
                files[f"Supplementary_Figure_S14_{k}"] = v
            removed_report_path = dirs.source_data / "step9_removed_panel_report.csv"
            pairwise_source_path = dirs.source_data / "step9_pairwise_similarity_matrix.csv"
            if removed_report_path.exists():
                files["step9_removed_panel_report"] = str(removed_report_path)
            if pairwise_source_path.exists():
                files["step9_pairwise_similarity_matrix"] = str(pairwise_source_path)
            checks.append(CheckResult("Supplementary_Figure_S14", "PASS", "Generated two-panel Supplementary Figure S14 with amino-acid enrichment and 3-mer enrichment; pairwise similarity retained as source data/report."))
        else:
            checks.append(CheckResult("Supplementary_Figure_S14", "WARN", "S14 not generated because reference/composition data were unavailable."))

    # Additional source-data summary outputs
    sim_summary_rows = []
    for context, col in [("Reference context", "nearest_reference_similarity"), ("Candidate context", "nearest_paper_candidate_similarity")]:
        arr = pd.to_numeric(final_df[col], errors="coerce").dropna().to_numpy(dtype=float)
        sim_summary_rows.append({"context": context, **percentile_summary(arr)})
    sim_summary = pd.DataFrame(sim_summary_rows)
    sim_summary.to_csv(dirs.source_data / "step9_contextual_similarity_summary.csv", index=False)
    files["step9_contextual_similarity_summary"] = str(dirs.source_data / "step9_contextual_similarity_summary.csv")

    arr = pairwise.to_numpy(dtype=float)
    mask = ~np.eye(arr.shape[0], dtype=bool)
    offdiag = arr[mask]
    bins = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 1.0]
    labels = ["0-0.05", "0.05-0.10", "0.10-0.15", "0.15-0.20", "0.20-0.30", "0.30-1.0"]
    cats = pd.cut(offdiag, bins=bins, labels=labels, include_lowest=True, right=False)
    div_summary = cats.value_counts().reindex(labels, fill_value=0).rename_axis("similarity_bin").reset_index(name="pair_count")
    div_summary["fraction"] = div_summary["pair_count"] / max(div_summary["pair_count"].sum(), 1)
    div_summary.to_csv(dirs.source_data / "step9_internal_diversity_summary.csv", index=False)
    files["step9_internal_diversity_summary"] = str(dirs.source_data / "step9_internal_diversity_summary.csv")

    try:
        comp_summary = residue_category_df(final_df, reference_df)
        comp_summary.to_csv(dirs.source_data / "step9_composition_context_summary.csv", index=False)
        files["step9_composition_context_summary"] = str(dirs.source_data / "step9_composition_context_summary.csv")
    except Exception:
        pass

    files = write_reports(dirs, checks, discovery, files, data.get("_paths", pd.DataFrame()))

    n_fail = sum(c.status == "FAIL" for c in checks)
    n_warn = sum(c.status == "WARN" for c in checks)
    status = "FAIL" if n_fail else ("WARN" if n_warn else "PASS")
    score = max(0, 100 - 15*n_fail - 2*n_warn)
    if not args.quiet:
        print("\nOncoPep Step 9 package generated.")
        print(f"Root: {dirs.root}")
        print(f"Readiness status: {status}; estimated score: {score}/100")
        print(f"Main figure: {files.get('Figure_5_png')}")
        if "Supplementary_Figure_S14_png" in files:
            print(f"Supplementary figure: {files.get('Supplementary_Figure_S14_png')}")
        print(f"Readiness report: {files.get('readiness_report')}")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
        if is_notebook() and rc != 0:
            raise RuntimeError(f"Step 9 script failed with exit code {rc}.")
        if not is_notebook():
            raise SystemExit(rc)
    except Exception as exc:
        if is_notebook():
            raise RuntimeError(str(exc)) from exc
        print("\nERROR: OncoPep Step 9 figure generation failed.\n", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(2)

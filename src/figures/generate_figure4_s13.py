#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OncoPep Step 8 — PLOS Computational Biology prioritization figure package.

SCRIPT_VERSION:
v8_2026_07_07_plos_step8_final_polish

Scientific role:
  Step 8 = multi-objective prioritization and prioritization robustness.
  This script does NOT repeat Step 7 descriptor-support or nearest-neighbor tail-risk
  figures. Descriptor distributions are exported only as source-data/report tables.

Generates:
  Fig 4. Multi-objective prioritization of generated OncoPep candidates.
  S13 Fig. Robustness of prioritization schemes.

Key v8 improvements:
  - Jupyter-safe argument parsing: no -f kernel.json contamination.
  - Safe role-aware discovery: no /run/user, kernel JSON, audit/count files.
  - Figure 4A: clearer prioritization-stage reduction with final-stage callout.
  - Figure 4B: includes condition fidelity if available; otherwise documents omission.
  - Figure 4B/C: IQR error bars are explicitly plotted and exported.
  - Figure 4C: n values are moved to x-axis labels, not inside bars.
  - S13A: Primary-first order, full scheme labels, diagonal de-emphasized.
  - S13B: denominator and percentages are added.
  - Full PLOS-style output package with source data, reports, manifest, README,
    requirements file, and code snapshot.
  - v8 polish: removes in-panel compression box from data region, improves metric
    tick labels, reduces panel-label dominance, and tightens S13 typography.

Recommended command:
  python OncoPep_step8_PLOS_prioritization_final.py \
    --step8-root /home/data3/Moe/nature_computational_peponco/PLOS/plos_comp/step_08 \
    --project-root /home/data3/Moe/nature_computational_peponco \
    --passed-file /home/data3/Moe/nature_computational_peponco/step8_v1/tables_supplementary/table_s8_12_full_ranked_passed_pool.csv \
    --shortlist-file /home/data3/Moe/nature_computational_peponco/step8_v1/tables_main/table_8_1_shortlist_main.csv \
    --stability-file /home/data3/Moe/nature_computational_peponco/step8_v1/tables_supplementary/table_s8_5_ranking_scheme_stability.csv \
    --recurrence-file /home/data3/Moe/nature_computational_peponco/step8_v1/tables_supplementary/table_s8_6_candidate_recurrence_across_schemes.csv \
    --generated-count 10840 --eligible-count 10291 --passed-count 10237 --shortlist-count 24 --final-count 12
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch


SCRIPT_NAME = "OncoPep_step8_PLOS_prioritization_final.py"
SCRIPT_VERSION = "v8_2026_07_07_plos_step8_final_polish"

DEFAULT_PROJECT_ROOT = Path("/home/data3/Moe/nature_computational_peponco")
DEFAULT_STEP8_ROOT = DEFAULT_PROJECT_ROOT / "PLOS" / "plos_comp" / "step_08"

SAFE_TABLE_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".pq"}
BLOCKED_EXTENSIONS = {".json", ".png", ".jpg", ".jpeg", ".pdf", ".tif", ".tiff", ".py", ".ipynb", ".md", ".log"}
BLOCKED_PATH_KEYWORDS = [
    "/run/user/",
    "/jupyter/runtime/",
    ".ipynb_checkpoints",
    "__pycache__",
    "/.git/",
    "kernel-",
]
BLOCKED_NAME_KEYWORDS = [
    "kernel",
    "runtime",
    "manifest",
    "readiness",
    "config",
    "summary_all",
    "source_data_all_panels",
    "panel_source",
    "split_count_validation_audit",
    "validation_audit",
    "count_audit",
    "readme",
    "requirements",
    "run_step8_debug",
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
    "amber": "#E69F00",
    "legend_bg": "#F7F7F7",
    "legend_edge": "#CCCCCC",
}

STAGE_COLORS = {
    "Generated": PLOS["light"],
    "QC-passed": PLOS["amber"],
    "Descriptor-plausible": PLOS["blue"],
    "Shortlist": PLOS["brown"],
    "Final panel": PLOS["coral"],
}

GROUP_COLORS = {
    "Descriptor-plausible": PLOS["blue"],
    "Shortlist": PLOS["brown"],
    "Final panel": PLOS["coral"],
}

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "oncopep_step8_overlap",
    [PLOS["white"], "#DDEFE4", PLOS["mint"], PLOS["blue"], PLOS["charcoal"]],
)
HEATMAP_CMAP.set_bad("#EFEFEF")

SCORE_ALIASES = {
    "novelty_score": [
        "novelty_score", "novelty", "exact_novelty_score", "novelty_component",
        "S_nov", "s_nov", "score_novelty", "nov_score"
    ],
    "descriptor_plausibility_score": [
        "descriptor_plausibility_score", "plausibility_score", "descriptor_plausibility",
        "reference_range_plausibility", "plausibility", "S_plaus", "s_plaus",
        "score_plausibility", "plaus_score"
    ],
    "condition_fidelity_score": [
        "condition_fidelity_score", "surrogate_condition_fidelity_score", "condition_score",
        "condition_fidelity", "condition_match_score", "condition_hit_rate",
        "condition_match_rate", "surrogate_condition_hit_rate", "condition_support_score",
        "S_cond", "s_cond", "score_condition", "cond_score"
    ],
    "diversity_score": [
        "diversity_score", "mmr_diversity_score", "diversity", "diversity_component",
        "S_div", "s_div", "score_diversity", "div_score"
    ],
    "final_score": [
        "final_score", "composite_score", "prioritization_score", "rank_score",
        "S_final", "s_final", "score_final", "total_score"
    ],
}

DESCRIPTOR_ALIASES = {
    "length": ["length", "peptide_length", "seq_len", "sequence_length"],
    "net_charge": ["net_charge", "net_charge_pH7", "net_charge_ph7", "charge", "charge_proxy", "net_charge_proxy"],
    "hydrophobicity": ["hydrophobicity", "mean_hydropathy", "mean_hydrophobicity", "hydropathy", "kd_hydrophobicity"],
    "entropy": ["entropy", "shannon_entropy", "sequence_entropy"],
}


@dataclass
class InputPaths:
    generated_file: Optional[Path] = None
    eligible_file: Optional[Path] = None
    passed_file: Optional[Path] = None
    shortlist_file: Optional[Path] = None
    final_file: Optional[Path] = None
    reference_file: Optional[Path] = None
    stability_file: Optional[Path] = None
    recurrence_file: Optional[Path] = None


@dataclass
class OutputDirs:
    root: Path
    main_figure: Path
    supplementary_figures: Path
    source_data: Path
    reports: Path
    code: Path


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


@dataclass
class DiscoveryRecord:
    role: str
    path: str
    decision: str
    reason: str
    score: int = 0


def is_jupyter_runtime_arg(x: str) -> bool:
    s = str(x)
    return (
        "/run/user/" in s
        or "/jupyter/runtime/" in s
        or "kernel-" in s
        or s.endswith(".json")
    )


def clean_argv(argv: Optional[Sequence[str]]) -> List[str]:
    raw = list(sys.argv[1:] if argv is None else argv)
    cleaned: List[str] = []
    skip_next = False

    for i, token in enumerate(raw):
        if skip_next:
            skip_next = False
            continue
        if token in {"-f", "--f", "--file"} and i + 1 < len(raw) and is_jupyter_runtime_arg(raw[i + 1]):
            skip_next = True
            continue
        if is_jupyter_runtime_arg(token):
            continue
        cleaned.append(token)

    return cleaned


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate OncoPep Step 8 PLOS prioritization figure/source-data package.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )

    parser.add_argument("--step8-root", type=Path, default=DEFAULT_STEP8_ROOT)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--search-root", action="append", type=Path, default=[])

    parser.add_argument("--generated-file", type=Path, default=None)
    parser.add_argument("--eligible-file", type=Path, default=None)
    parser.add_argument("--passed-file", type=Path, default=None)
    parser.add_argument("--shortlist-file", type=Path, default=None)
    parser.add_argument("--final-file", type=Path, default=None)
    parser.add_argument("--reference-file", type=Path, default=None)
    parser.add_argument("--stability-file", type=Path, default=None)
    parser.add_argument("--recurrence-file", type=Path, default=None)

    parser.add_argument("--generated-count", type=int, default=None)
    parser.add_argument("--eligible-count", type=int, default=None)
    parser.add_argument("--passed-count", type=int, default=None)
    parser.add_argument("--shortlist-count", type=int, default=None)
    parser.add_argument("--final-count", type=int, default=None)

    parser.add_argument("--png-dpi", type=int, default=300)
    parser.add_argument("--tiff-dpi", type=int, default=600)
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-notebook-results", action="store_true")
    parser.add_argument("--auto-discover", action="store_true", help="Enable safe auto-discovery for optional missing files.")
    parser.add_argument("--demo-data", action="store_true", help="Use synthetic demo data. Never use for manuscript output.")

    cleaned = clean_argv(argv)
    args, unknown = parser.parse_known_args(cleaned)
    args.unknown_args_ignored = unknown
    return args


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
        Path(d).mkdir(parents=True, exist_ok=True)
    return dirs


def path_is_blocked(path: Path) -> Tuple[bool, str]:
    s = str(path)
    sl = s.lower()
    name = path.name.lower()

    if path.suffix.lower() in BLOCKED_EXTENSIONS:
        return True, f"blocked extension {path.suffix}"

    for kw in BLOCKED_PATH_KEYWORDS:
        if kw.lower() in sl:
            return True, f"blocked path keyword {kw}"

    for kw in BLOCKED_NAME_KEYWORDS:
        if kw.lower() in name:
            return True, f"blocked name keyword {kw}"

    if path.suffix.lower() not in SAFE_TABLE_EXTENSIONS:
        return True, f"not an allowed table extension: {path.suffix}"

    return False, "allowed"


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    blocked, reason = path_is_blocked(path)
    if blocked:
        raise ValueError(f"Rejected unsafe/non-table input: {path} ({reason})")
    if not path.exists():
        raise FileNotFoundError(f"Input table does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        try:
            return pd.read_csv(path, sep="\t")
        except Exception:
            return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported table format: {path}")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def find_first_col(df: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    lower_to_original = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a in df.columns:
            return a
        if a.lower() in lower_to_original:
            return lower_to_original[a.lower()]
    return None


def find_numeric_condition_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        cl = c.lower()
        if "condition" in cl and any(x in cl for x in ["score", "fidelity", "match", "hit", "support"]):
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().any():
                cols.append(c)
    return cols


def table_looks_candidate_like(df: pd.DataFrame, role: str) -> Tuple[bool, str]:
    if df is None or len(df) == 0:
        return False, "zero rows"

    cols = {c.lower(): c for c in df.columns}
    row_count = len(df)

    peptide_like = any(c in cols for c in ["sequence", "peptide", "peptide_sequence", "seq"])
    score_like = any(
        c in cols for c in [
            "final_score", "s_final", "novelty_score", "descriptor_plausibility_score",
            "condition_fidelity_score", "diversity_score", "rank", "rank_score"
        ]
    )

    if role in {"passed", "shortlist", "final"}:
        if not peptide_like and not score_like:
            return False, "no peptide-like or score-like columns"
        if role == "final" and row_count > 500:
            return False, "too many rows for final panel"
        return True, "candidate-like table"

    if role in {"generated", "eligible"}:
        if row_count < 10:
            return False, "too few rows for generated/QC-passed table"
        return True, "large candidate-stage table"

    return True, "accepted"


def score_file_for_role(path: Path, role: str) -> int:
    name = path.name.lower()
    full = str(path).lower()
    score = 0

    preferred_dirs = [
        "step8_v1/tables_main",
        "step8_v1/tables_supplementary",
        "step8/tables_main",
        "step8/tables_supplementary",
        "step6_v5/tables",
        "step2/tables",
    ]
    for d in preferred_dirs:
        if d in full:
            score += 12

    role_keywords = {
        "generated": ["all_generated", "generated_sequences", "generated"],
        "eligible": ["qc_passed", "eligible", "valid_candidates"],
        "passed": ["full_ranked_passed_pool", "passed_pool", "descriptor_plausible", "plausible"],
        "shortlist": ["shortlist", "short_list", "top24", "table_8_1"],
        "final": ["final_panel", "final_candidates", "final_12"],
        "reference": ["train_only_novelty_reference", "train_reference", "reference"],
        "stability": ["ranking_scheme_stability", "scheme_stability", "scheme_overlap"],
        "recurrence": ["candidate_recurrence", "recurrence_across_schemes"],
    }

    for kw in role_keywords.get(role, []):
        if kw in name:
            score += 20

    negative = ["split", "count", "audit", "validation", "readiness", "manifest", "source_data_all"]
    if role in {"generated", "eligible", "passed", "shortlist", "final"}:
        for kw in negative:
            if kw in name:
                score -= 100

    return score


def safe_discover_file(role: str, explicit: Optional[Path], roots: Sequence[Path]) -> Tuple[Optional[Path], List[DiscoveryRecord]]:
    records: List[DiscoveryRecord] = []

    if explicit is not None:
        p = Path(explicit)
        blocked, reason = path_is_blocked(p)
        if blocked:
            records.append(DiscoveryRecord(role, str(p), "REJECT", f"explicit path rejected: {reason}", -999))
            return None, records
        records.append(DiscoveryRecord(role, str(p), "SELECT", "explicit path supplied", 999))
        return p, records

    patterns = {
        "generated": ["*generated*.csv", "*all_generated*.csv"],
        "eligible": ["*qc*pass*.csv", "*eligible*.csv", "*valid*.csv"],
        "passed": ["*full_ranked_passed_pool*.csv", "*passed_pool*.csv", "*descriptor*plaus*.csv", "*plausible*.csv"],
        "shortlist": ["*shortlist*.csv", "*table_8_1*.csv"],
        "final": ["*final_panel*.csv", "*final_candidates*.csv", "*final_12*.csv"],
        "reference": ["*train_only_novelty_reference*.csv", "*reference*.csv"],
        "stability": ["*ranking_scheme_stability*.csv", "*scheme*stability*.csv", "*scheme*overlap*.csv"],
        "recurrence": ["*candidate_recurrence*.csv", "*recurrence*.csv"],
    }

    candidates: List[Tuple[int, Path]] = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            records.append(DiscoveryRecord(role, str(root), "SKIP", "root does not exist", 0))
            continue
        if any(kw in str(root).lower() for kw in ["/run/user/", "/jupyter/runtime/", ".ipynb_checkpoints", "__pycache__", "/.git/"]):
            records.append(DiscoveryRecord(role, str(root), "SKIP", "unsafe root", -999))
            continue

        for pat in patterns.get(role, []):
            for p in root.rglob(pat):
                if not p.is_file():
                    continue
                blocked, reason = path_is_blocked(p)
                if blocked:
                    records.append(DiscoveryRecord(role, str(p), "REJECT", reason, -999))
                    continue
                score = score_file_for_role(p, role)
                if score <= 0:
                    records.append(DiscoveryRecord(role, str(p), "REJECT", "low role score", score))
                    continue
                records.append(DiscoveryRecord(role, str(p), "CANDIDATE", "safe candidate", score))
                candidates.append((score, p))

    if not candidates:
        return None, records

    candidates = sorted(candidates, key=lambda x: (x[0], -len(str(x[1]))), reverse=True)

    for score, p in candidates[:12]:
        try:
            preview = read_table(p)
            ok, reason = table_looks_candidate_like(preview, role)
            if ok:
                records.append(DiscoveryRecord(role, str(p), "SELECT", reason, score))
                return p, records
            records.append(DiscoveryRecord(role, str(p), "REJECT", reason, score))
        except Exception as e:
            records.append(DiscoveryRecord(role, str(p), "REJECT", f"cannot read preview: {e}", score))

    return None, records


def discover_inputs(args: argparse.Namespace) -> Tuple[InputPaths, List[DiscoveryRecord]]:
    roots = [
        args.project_root / "step8_v1" / "tables_main",
        args.project_root / "step8_v1" / "tables_supplementary",
        args.project_root / "step6_v5" / "tables",
        args.project_root / "step2" / "tables",
        args.step8_root,
    ] + list(args.search_root)

    records: List[DiscoveryRecord] = []

    def get(role: str, explicit: Optional[Path]) -> Optional[Path]:
        p, rec = safe_discover_file(role, explicit, roots if args.auto_discover or explicit is not None else [])
        records.extend(rec)
        return p

    inputs = InputPaths(
        generated_file=get("generated", args.generated_file),
        eligible_file=get("eligible", args.eligible_file),
        passed_file=get("passed", args.passed_file),
        shortlist_file=get("shortlist", args.shortlist_file),
        final_file=get("final", args.final_file),
        reference_file=get("reference", args.reference_file),
        stability_file=get("stability", args.stability_file),
        recurrence_file=get("recurrence", args.recurrence_file),
    )
    return inputs, records


def load_optional_table(path: Optional[Path], key: str, checks: List[CheckResult]) -> Optional[pd.DataFrame]:
    if path is None:
        checks.append(CheckResult(key, "INFO", "No input path supplied or discovered."))
        return None
    try:
        df = read_table(path)
        checks.append(CheckResult(key, "PASS", f"Loaded {len(df):,} rows from {path}"))
        return df
    except Exception as e:
        checks.append(CheckResult(key, "FAIL", f"Could not read {path}: {e}"))
        return None



def make_demo_data(seed: int = 42) -> Dict[str, pd.DataFrame]:
    """Synthetic data for script testing only. Never use demo outputs for manuscript."""
    rng = np.random.default_rng(seed)

    def mk(n: int, stage: str, shift: float) -> pd.DataFrame:
        return pd.DataFrame({
            "sequence": [f"ACD{stage[:2].upper()}{i:05d}KLMNPQ" for i in range(n)],
            "stage": stage,
            "novelty_score": np.clip(rng.normal(0.94 + shift, 0.025, n), 0, 1),
            "descriptor_plausibility_score": np.clip(rng.normal(0.60 + 1.0 * shift, 0.09, n), 0, 1),
            "condition_fidelity_score": np.clip(rng.normal(0.72 + 0.55 * shift, 0.08, n), 0, 1),
            "diversity_score": np.clip(rng.normal(0.88 + 0.25 * shift, 0.035, n), 0, 1),
            "final_score": np.clip(rng.normal(0.74 + 0.65 * shift, 0.065, n), 0, 1),
            "length": np.clip(rng.normal(35 + 4 * shift, 8, n), 5, 60),
            "net_charge": rng.normal(3 + 2 * shift, 3, n),
            "hydrophobicity": rng.normal(0 + 0.2 * shift, 0.8, n),
            "entropy": np.clip(rng.normal(3.3 + 0.2 * shift, 0.4, n), 0, 4.2),
            "rank": np.arange(1, n + 1),
        })

    schemes = ["Primary", "Novelty-heavy", "Plausibility-heavy", "Diversity-heavy"]
    stability = []
    for i, a in enumerate(schemes):
        for j, b in enumerate(schemes):
            if i < j:
                stability.append({"scheme_a": a, "scheme_b": b, "jaccard_overlap": float(rng.uniform(0.32, 0.72))})

    recurrence = pd.DataFrame({
        "candidate_id": [f"C{i:02d}" for i in range(1, 41)],
        "scheme_recurrence_n": rng.choice([1, 2, 3, 4], size=40, p=[0.34, 0.18, 0.22, 0.26]),
    })

    return {
        "gen_df": mk(10840, "generated", -0.10),
        "eligible_df": mk(10291, "eligible", -0.06),
        "passed_df": mk(10237, "passed", 0.00),
        "shortlist_df": mk(24, "shortlist", 0.12),
        "final_panel_df": mk(12, "final", 0.16),
        "ref_df": mk(1000, "reference", -0.02),
        "stability_df": pd.DataFrame(stability),
        "recurrence_df": recurrence,
    }


def load_inputs(args: argparse.Namespace, inputs: InputPaths) -> Tuple[Dict[str, pd.DataFrame], List[CheckResult]]:
    checks: List[CheckResult] = []
    data: Dict[str, pd.DataFrame] = {}

    mapping = {
        "generated_file": "gen_df",
        "eligible_file": "eligible_df",
        "passed_file": "passed_df",
        "shortlist_file": "shortlist_df",
        "final_file": "final_panel_df",
        "reference_file": "ref_df",
        "stability_file": "stability_df",
        "recurrence_file": "recurrence_df",
    }

    for attr, key in mapping.items():
        df = load_optional_table(getattr(inputs, attr), key, checks)
        if df is not None:
            data[key] = df

    if args.demo_data and ("passed_df" not in data or "shortlist_df" not in data):
        demo = make_demo_data(args.seed)
        for k, v in demo.items():
            if k not in data:
                data[k] = v
        checks.append(CheckResult("demo_data", "WARN", "Synthetic demo data were generated. Do not use demo output for manuscript."))

    if "passed_df" not in data:
        raise RuntimeError("Required passed/descriptor-plausible pool is missing. Supply --passed-file.")
    if "shortlist_df" not in data:
        raise RuntimeError("Required shortlist table is missing. Supply --shortlist-file.")

    if "final_panel_df" not in data:
        final_n = args.final_count or 12
        shortlist = order_candidate_table(data["shortlist_df"].copy())
        if len(shortlist) >= final_n:
            data["final_panel_df"] = shortlist.head(final_n).copy()
            checks.append(CheckResult("final_panel_df", "INFO", f"No final-file supplied; inferred final panel from first {final_n} ranked shortlist rows."))
        else:
            raise RuntimeError("Final panel missing and shortlist has fewer rows than requested final-count. Supply --final-file.")

    if "eligible_df" not in data:
        data["eligible_df"] = data["passed_df"].copy()
        if args.eligible_count is not None:
            checks.append(CheckResult("eligible_df", "INFO", "No QC-passed table supplied; count override used for Fig 4A and passed_df used only for row-level placeholder summaries."))
        else:
            checks.append(CheckResult("eligible_df", "WARN", "No valid QC-passed table supplied; using passed_df for row-level summaries."))

    if "gen_df" not in data:
        data["gen_df"] = data["passed_df"].copy()
        if args.generated_count is not None:
            checks.append(CheckResult("gen_df", "INFO", "No generated table supplied; count override used for Fig 4A and passed_df used only for row-level placeholder summaries."))
        else:
            checks.append(CheckResult("gen_df", "WARN", "No generated table supplied; using passed_df for row-level summaries."))

    return data, checks


def order_candidate_table(df: pd.DataFrame) -> pd.DataFrame:
    """Use stable ranking when available; otherwise preserve input order."""
    rank_col = find_first_col(df, ["rank", "ranking", "primary_rank", "final_rank", "selection_rank"])
    if rank_col is not None:
        tmp = df.copy()
        tmp["_rank_tmp"] = pd.to_numeric(tmp[rank_col], errors="coerce")
        tmp = tmp.sort_values(["_rank_tmp"], na_position="last").drop(columns=["_rank_tmp"])
        return tmp

    score_col = find_first_col(df, SCORE_ALIASES["final_score"])
    if score_col is not None:
        tmp = df.copy()
        tmp["_score_tmp"] = pd.to_numeric(tmp[score_col], errors="coerce")
        tmp = tmp.sort_values(["_score_tmp"], ascending=False, na_position="last").drop(columns=["_score_tmp"])
        return tmp

    return df


def clean_numeric(series: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def median_iqr(arr: np.ndarray) -> Tuple[float, float, float, int]:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    return float(med), float(q1), float(q3), int(n)


def bootstrap_ci(arr: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        stats[i] = np.median(rng.choice(arr, size=len(arr), replace=True))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def ensure_score_columns(data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], List[CheckResult]]:
    checks: List[CheckResult] = []
    component_cols = ["novelty_score", "descriptor_plausibility_score", "condition_fidelity_score", "diversity_score"]

    for key in ["passed_df", "shortlist_df", "final_panel_df"]:
        df = data[key].copy()

        for canon, aliases in SCORE_ALIASES.items():
            if canon in df.columns:
                df[canon] = pd.to_numeric(df[canon], errors="coerce")
                continue
            found = find_first_col(df, aliases)
            if found is not None:
                df[canon] = pd.to_numeric(df[found], errors="coerce")
                checks.append(CheckResult(f"{key}:{canon}", "PASS", f"Mapped `{found}` to `{canon}`."))
            elif canon == "condition_fidelity_score":
                condition_cols = find_numeric_condition_columns(df)
                if condition_cols:
                    df[canon] = df[condition_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                    checks.append(CheckResult(f"{key}:{canon}", "PASS", f"Computed condition_fidelity_score from columns: {condition_cols}."))
                else:
                    checks.append(CheckResult(f"{key}:{canon}", "INFO", "No condition-fidelity column found; metric omitted unless unavailable in all stages."))
            else:
                checks.append(CheckResult(f"{key}:{canon}", "INFO", f"No column found for `{canon}`."))

        available = [c for c in component_cols if c in df.columns and df[c].notna().any()]
        if "final_score" not in df.columns or not df["final_score"].notna().any():
            if available:
                df["final_score"] = df[available].mean(axis=1)
                checks.append(CheckResult(f"{key}:final_score", "INFO", f"Computed final_score as mean of available components: {available}."))
            elif "rank" in df.columns:
                rank = pd.to_numeric(df["rank"], errors="coerce")
                denom = max(float(rank.max() - rank.min()), 1.0)
                df["final_score"] = 1.0 - ((rank - rank.min()) / denom)
                checks.append(CheckResult(f"{key}:final_score", "INFO", "Computed final_score from rank."))
            else:
                df["final_score"] = np.linspace(1.0, 0.5, len(df))
                checks.append(CheckResult(f"{key}:final_score", "WARN", "No score columns found; assigned monotonic placeholder final_score for plotting only."))

        data[key] = df

    return data, checks


def ensure_descriptor_columns(data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], List[CheckResult]]:
    checks: List[CheckResult] = []

    for key, df in list(data.items()):
        if not isinstance(df, pd.DataFrame):
            continue
        df2 = df.copy()
        for canon, aliases in DESCRIPTOR_ALIASES.items():
            if canon in df2.columns:
                df2[canon] = pd.to_numeric(df2[canon], errors="coerce")
                continue
            found = find_first_col(df2, aliases)
            if found is not None:
                df2[canon] = pd.to_numeric(df2[found], errors="coerce")
                checks.append(CheckResult(f"{key}:{canon}", "PASS", f"Mapped `{found}` to `{canon}`."))
        data[key] = df2

    return data, checks


def available_component_metrics(data: Dict[str, pd.DataFrame]) -> List[Tuple[str, str]]:
    ordered = [
        ("novelty_score", "Novelty"),
        ("descriptor_plausibility_score", "Plausibility"),
        ("condition_fidelity_score", "Condition fidelity"),
        ("diversity_score", "Diversity"),
    ]

    out = []
    for col, lab in ordered:
        ok = all(col in data[k].columns and data[k][col].notna().any() for k in ["passed_df", "shortlist_df", "final_panel_df"])
        if ok:
            out.append((col, lab))

    if not out:
        out = [("final_score", "Composite score")]

    return out


def summarize_scores(data: Dict[str, pd.DataFrame], metrics: Sequence[Tuple[str, str]], args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    stages = [("Descriptor-plausible", "passed_df"), ("Shortlist", "shortlist_df"), ("Final panel", "final_panel_df")]

    for stage_label, key in stages:
        df = data[key]
        for metric_col, metric_label in metrics:
            arr = clean_numeric(df[metric_col])
            med, q1, q3, n = median_iqr(arr)
            lo, hi = bootstrap_ci(arr, args.bootstrap_n, args.seed + len(rows))
            rows.append({
                "stage": stage_label,
                "metric": metric_label,
                "metric_column": metric_col,
                "n": n,
                "median": med,
                "q1": q1,
                "q3": q3,
                "iqr_low": q1,
                "iqr_high": q3,
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
                "plotted_error_type": "IQR",
            })

    return pd.DataFrame(rows)


def final_score_summary(data: Dict[str, pd.DataFrame], args: argparse.Namespace) -> pd.DataFrame:
    return summarize_scores(data, [("final_score", "Composite score")], args)


def stage_count_table(data: Dict[str, pd.DataFrame], args: argparse.Namespace) -> pd.DataFrame:
    generated_n = args.generated_count if args.generated_count is not None else len(data["gen_df"])
    eligible_n = args.eligible_count if args.eligible_count is not None else len(data["eligible_df"])
    passed_n = args.passed_count if args.passed_count is not None else len(data["passed_df"])
    shortlist_n = args.shortlist_count if args.shortlist_count is not None else len(data["shortlist_df"])
    final_n = args.final_count if args.final_count is not None else len(data["final_panel_df"])

    rows = [
        ("Generated", "All generated OncoPep outputs", generated_n, "explicit_override" if args.generated_count is not None else "table_length"),
        ("QC-passed", "Generated outputs passing syntax/QC filters", eligible_n, "explicit_override" if args.eligible_count is not None else "table_length"),
        ("Descriptor-plausible", "QC-passed outputs retained by descriptor plausibility filtering", passed_n, "explicit_override" if args.passed_count is not None else "table_length"),
        ("Shortlist", "Diversity-aware shortlist", shortlist_n, "explicit_override" if args.shortlist_count is not None else "table_length"),
        ("Final panel", "Final diversity-enriched panel", final_n, "explicit_override" if args.final_count is not None else "table_length"),
    ]

    out = pd.DataFrame(rows, columns=["stage", "definition", "count", "count_source"])
    out["fraction_of_generated"] = out["count"] / generated_n if generated_n else np.nan
    out["percent_of_generated"] = out["fraction_of_generated"] * 100.0
    return out


def descriptor_distribution_summary(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    stages = [
        ("Descriptor-plausible", "passed_df"),
        ("Shortlist", "shortlist_df"),
        ("Final panel", "final_panel_df"),
        ("Reference", "ref_df"),
    ]

    for stage, key in stages:
        if key not in data:
            continue
        df = data[key]
        for desc, label in [
            ("length", "Peptide length"),
            ("net_charge", "Net charge at pH 7"),
            ("hydrophobicity", "Mean hydrophobicity"),
            ("entropy", "Shannon entropy"),
        ]:
            if desc not in df.columns:
                continue
            arr = clean_numeric(df[desc])
            med, q1, q3, n = median_iqr(arr)
            rows.append({
                "stage": stage,
                "descriptor": label,
                "descriptor_column": desc,
                "n": n,
                "median": med,
                "q1": q1,
                "q3": q3,
                "min": float(np.min(arr)) if len(arr) else np.nan,
                "max": float(np.max(arr)) if len(arr) else np.nan,
                "mean": float(np.mean(arr)) if len(arr) else np.nan,
                "sd": float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan,
            })

    return pd.DataFrame(rows)


def normalize_scheme_label(label: Any) -> str:
    s = str(label).strip().replace("_", " ").replace("-", " ")
    sl = s.lower()
    if "primary" in sl:
        return "Primary"
    if "novel" in sl:
        return "Novelty-weighted"
    if "plaus" in sl:
        return "Plausibility-weighted"
    if "divers" in sl:
        return "Diversity-weighted"
    if "condition" in sl or "cond" in sl:
        return "Condition-weighted"
    return " ".join(w.capitalize() for w in s.split())


def scheme_order(labels: Sequence[str]) -> List[str]:
    priority = ["Primary", "Novelty-weighted", "Plausibility-weighted", "Condition-weighted", "Diversity-weighted"]
    labels_unique = []
    for x in labels:
        if x not in labels_unique:
            labels_unique.append(x)
    ordered = [x for x in priority if x in labels_unique]
    ordered += sorted([x for x in labels_unique if x not in ordered])
    return ordered


def stability_matrix_source(stability_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    needed = {"scheme_a", "scheme_b", "jaccard_overlap"}
    if stability_df is None or len(stability_df) == 0 or not needed.issubset(stability_df.columns):
        return pd.DataFrame(), pd.DataFrame()

    df = stability_df.copy()
    df["scheme_a_label"] = df["scheme_a"].map(normalize_scheme_label)
    df["scheme_b_label"] = df["scheme_b"].map(normalize_scheme_label)
    names = scheme_order(list(df["scheme_a_label"]) + list(df["scheme_b_label"]))

    mat = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for _, r in df.iterrows():
        a = str(r["scheme_a_label"])
        b = str(r["scheme_b_label"])
        v = float(r["jaccard_overlap"])
        mat.loc[a, b] = v
        mat.loc[b, a] = v

    long = mat.reset_index(names="scheme_a").melt(
        id_vars="scheme_a",
        var_name="scheme_b",
        value_name="jaccard_overlap",
    )
    long["is_diagonal"] = long["scheme_a"] == long["scheme_b"]
    long["interpretation"] = np.where(long["is_diagonal"], "self-overlap", "off-diagonal robustness overlap")
    return mat, long


def recurrence_source(recurrence_df: pd.DataFrame) -> pd.DataFrame:
    if recurrence_df is None or len(recurrence_df) == 0 or "scheme_recurrence_n" not in recurrence_df.columns:
        return pd.DataFrame(columns=["scheme_recurrence_n", "candidate_count", "fraction", "percent", "denominator_n"])

    vals = pd.to_numeric(recurrence_df["scheme_recurrence_n"], errors="coerce").dropna().astype(int)
    counts = vals.value_counts().sort_index()
    total = int(counts.sum())
    out = pd.DataFrame({
        "scheme_recurrence_n": counts.index,
        "candidate_count": counts.values,
    })
    out["denominator_n"] = total
    out["fraction"] = out["candidate_count"] / total if total else np.nan
    out["percent"] = out["fraction"] * 100.0
    return out


def combine_source_data(named_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    all_cols: List[str] = []

    for name, df in named_tables.items():
        if df is None or len(df) == 0:
            continue
        tmp = df.copy()
        tmp.insert(0, "source_table", name)
        frames.append(tmp)
        for c in tmp.columns:
            if c not in all_cols:
                all_cols.append(c)

    if not frames:
        return pd.DataFrame()

    return pd.concat([f.reindex(columns=all_cols) for f in frames], ignore_index=True)


def set_publication_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": PLOS["white"],
        "savefig.facecolor": PLOS["white"],
        "axes.facecolor": PLOS["white"],
        "font.family": "DejaVu Sans",
        "font.size": 8.2,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.8,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 8.2,
        "axes.edgecolor": "#B8B8B8",
        "axes.labelcolor": PLOS["dark"],
        "xtick.color": PLOS["dark"],
        "ytick.color": PLOS["dark"],
        "text.color": PLOS["dark"],
        "axes.titlecolor": PLOS["dark"],
        "axes.grid": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_axis(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.set_axisbelow(True)
    if grid_axis == "x":
        ax.xaxis.grid(True, color=PLOS["grid"], linewidth=0.75)
    elif grid_axis == "y":
        ax.yaxis.grid(True, color=PLOS["grid"], linewidth=0.75)
    elif grid_axis == "both":
        ax.grid(True, color=PLOS["grid"], linewidth=0.75)

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#B8B8B8")
        ax.spines[side].set_linewidth(0.85)
    ax.tick_params(width=0.75, length=3.3, colors=PLOS["dark"])


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.03, size: float = 14) -> None:
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=size,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=PLOS["dark"],
        clip_on=False,
    )


def save_figure(fig: plt.Figure, out_base: Path, args: argparse.Namespace) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=args.png_dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".tiff"), dpi=args.tiff_dpi, bbox_inches="tight")
    plt.close(fig)


def stage_xtick_label(stage: str, n: int) -> str:
    if stage == "Descriptor-plausible":
        return f"Descriptor-\nplausible\nn={n:,}"
    if stage == "Final panel":
        return f"Final\npanel\nn={n:,}"
    return f"{stage}\nn={n:,}"



def metric_xtick_label(label: str) -> str:
    mapping = {
        "Novelty": "Novelty",
        "Plausibility": "Plausibility",
        "Condition fidelity": "Condition\nfidelity",
        "Diversity": "Diversity",
        "Composite score": "Composite\nscore",
    }
    return mapping.get(str(label), str(label).replace(" ", "\n"))

def plot_figure4(
    counts: pd.DataFrame,
    support_summary: pd.DataFrame,
    final_score_sum: pd.DataFrame,
    out_base: Path,
    args: argparse.Namespace,
) -> None:
    set_publication_style()
    fig = plt.figure(figsize=(13.9, 4.9))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.30, 1.46, 1.02], wspace=0.50)

    # Panel A
    ax = fig.add_subplot(gs[0, 0])
    add_panel_label(ax, "A", x=-0.17, size=14)
    df_plot = counts.iloc[::-1].copy()
    y = np.arange(len(df_plot))
    colors = [STAGE_COLORS.get(s, PLOS["light"]) for s in df_plot["stage"]]
    ax.barh(y, df_plot["count"], color=colors, edgecolor="none", height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(df_plot["stage"])
    ax.set_xlabel("Number of peptides (log scale)")
    ax.set_title("Prioritization-stage reduction", pad=8)
    ax.set_xscale("log")

    xmin = max(1, float(df_plot["count"].min()) * 0.65)
    xmax = float(df_plot["count"].max()) * 2.05
    ax.set_xlim(xmin, xmax)
    style_axis(ax, grid_axis="x")

    for yi, (_, r) in zip(y, df_plot.iterrows()):
        weight = "bold" if r["stage"] in {"Shortlist", "Final panel"} else "normal"
        ax.text(
            int(r["count"]) * 1.08, yi,
            f"{int(r['count']):,} ({float(r['percent_of_generated']):.1f}%)",
            va="center", ha="left", fontsize=8.0, fontweight=weight,
        )

    # Keep the prioritization-compression interpretation in source data/report/caption,
    # not as a large in-panel box, to avoid duplicating the Step 7 audit/funnel.
    try:
        plausible_n = int(counts.loc[counts["stage"] == "Descriptor-plausible", "count"].iloc[0])
        shortlist_n = int(counts.loc[counts["stage"] == "Shortlist", "count"].iloc[0])
        final_n = int(counts.loc[counts["stage"] == "Final panel", "count"].iloc[0])
        ax.text(
            0.02, -0.22,
            f"Late-stage prioritization compression: {plausible_n:,} → {shortlist_n:,} → {final_n:,}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.1,
            color=PLOS["dark"],
            clip_on=False,
        )
    except Exception:
        pass

    # Panel B
    ax = fig.add_subplot(gs[0, 1])
    add_panel_label(ax, "B", x=-0.11, size=14)
    metrics = list(support_summary["metric"].drop_duplicates())
    stages = ["Descriptor-plausible", "Shortlist", "Final panel"]
    x = np.arange(len(metrics))
    width = min(0.22, 0.74 / max(len(stages), 1))

    for i, stage in enumerate(stages):
        sub = support_summary[support_summary["stage"] == stage].set_index("metric").reindex(metrics).reset_index()
        xpos = x + (i - 1) * width
        vals = sub["median"].to_numpy(float)
        q1 = sub["q1"].to_numpy(float)
        q3 = sub["q3"].to_numpy(float)
        yerr = np.vstack([vals - q1, q3 - vals])
        ax.bar(xpos, vals, width=width, color=GROUP_COLORS[stage], edgecolor="none", label=stage, zorder=3)
        ax.errorbar(
            xpos, vals, yerr=yerr,
            fmt="none", ecolor=PLOS["dark"], elinewidth=0.7,
            capsize=2.0, capthick=0.7, zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([metric_xtick_label(m) for m in metrics], rotation=0, ha="center")
    ax.tick_params(axis="x", pad=3)
    ax.set_ylabel("Median score (IQR)")
    ax.set_title("Multi-objective support scores", pad=8)
    ax.set_ylim(0, 1.04)
    style_axis(ax, grid_axis="y")

    # Panel C
    ax = fig.add_subplot(gs[0, 2])
    add_panel_label(ax, "C", x=-0.17, size=14)
    sub = final_score_sum.set_index("stage").reindex(stages).reset_index()
    xpos = np.arange(len(stages))
    vals = sub["median"].to_numpy(float)
    q1 = sub["q1"].to_numpy(float)
    q3 = sub["q3"].to_numpy(float)
    yerr = np.vstack([vals - q1, q3 - vals])
    bars = ax.bar(xpos, vals, color=[GROUP_COLORS[s] for s in stages], edgecolor="none", width=0.58, zorder=3)
    ax.errorbar(
        xpos, vals, yerr=yerr,
        fmt="none", ecolor=PLOS["dark"], elinewidth=0.75,
        capsize=2.2, capthick=0.75, zorder=4,
    )
    xlabels = [stage_xtick_label(stage, int(n)) for stage, n in zip(sub["stage"], sub["n"].fillna(0).astype(int))]
    ax.set_xticks(xpos)
    ax.set_xticklabels(xlabels)
    ax.tick_params(axis="x", labelsize=7.2, pad=3)
    ax.set_ylabel("Median composite score (IQR)")
    ax.set_title("Composite-score enrichment", pad=8)
    ax.set_ylim(0, 1.04)
    style_axis(ax, grid_axis="y")

    handles = [Patch(facecolor=GROUP_COLORS[s], edgecolor="none", label=s) for s in stages]
    leg = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.58, 0.010),
        ncol=3,
        frameon=True,
        columnspacing=1.1,
        handlelength=1.4,
        borderpad=0.35,
    )
    leg.get_frame().set_facecolor(PLOS["legend_bg"])
    leg.get_frame().set_edgecolor(PLOS["legend_edge"])
    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.30)
    save_figure(fig, out_base, args)


def label_for_heatmap(label: str) -> str:
    mapping = {
        "Primary": "Primary",
        "Novelty-weighted": "Novelty-\nweighted",
        "Plausibility-weighted": "Plausibility-\nweighted",
        "Condition-weighted": "Condition-\nweighted",
        "Diversity-weighted": "Diversity-\nweighted",
    }
    return mapping.get(label, label.replace(" ", "\n"))


def plot_s13(
    stability_mat: pd.DataFrame,
    recurrence_counts: pd.DataFrame,
    out_base: Path,
    args: argparse.Namespace,
) -> None:
    set_publication_style()
    fig = plt.figure(figsize=(8.8, 4.35))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.16, 0.96], wspace=0.42)

    # S13A
    ax = fig.add_subplot(gs[0, 0])
    add_panel_label(ax, "A", x=-0.15, size=14)
    if len(stability_mat) > 0:
        plot_mat = stability_mat.astype(float).copy()
        np.fill_diagonal(plot_mat.values, np.nan)
        im = ax.imshow(plot_mat.values, vmin=0, vmax=1, cmap=HEATMAP_CMAP)

        labels = [label_for_heatmap(x) for x in stability_mat.index.tolist()]
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=0, ha="center")
        ax.set_yticklabels(labels)
        ax.set_title("Ranking-scheme overlap", pad=8)
        ax.grid(False)

        for i in range(stability_mat.shape[0]):
            for j in range(stability_mat.shape[1]):
                val = float(stability_mat.values[i, j])
                if i == j:
                    ax.text(j, i, "self", ha="center", va="center", fontsize=6.6, color="#777777")
                else:
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.2, color=PLOS["white"] if val >= 0.62 else PLOS["dark"])

        for spine in ax.spines.values():
            spine.set_color("#B8B8B8")
            spine.set_linewidth(0.8)
        cbar = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.030)
        cbar.set_label("Jaccard overlap")
        cbar.ax.tick_params(labelsize=7.5, length=3)
        cbar.outline.set_edgecolor("#B8B8B8")
    else:
        ax.text(0.5, 0.5, "Scheme-overlap data unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

    # S13B
    ax = fig.add_subplot(gs[0, 1])
    add_panel_label(ax, "B", x=-0.15, size=14)
    if len(recurrence_counts) > 0:
        x = recurrence_counts["scheme_recurrence_n"].astype(int).to_numpy()
        y = recurrence_counts["candidate_count"].astype(int).to_numpy()
        pct = recurrence_counts["percent"].to_numpy(float)
        denom = int(recurrence_counts["denominator_n"].iloc[0])
        bars = ax.bar(x, y, color=PLOS["brown"], edgecolor="none", width=0.62, zorder=3)
        ax.set_xlabel("Schemes recovering candidate")
        ax.set_ylabel("Candidate count")
        ax.set_title("Candidate recurrence", pad=8)
        ax.set_xticks(x)
        ax.set_ylim(0, max(y) * 1.25 if len(y) and max(y) > 0 else 1)
        style_axis(ax, grid_axis="y")
        for b, val, p in zip(bars, y, pct):
            ax.text(
                b.get_x() + b.get_width() / 2,
                val + ax.get_ylim()[1] * 0.025,
                f"{val}\n({p:.0f}%)",
                ha="center",
                va="bottom",
                fontsize=7.4,
            )
        ax.text(
            0.98, 0.96,
            f"n = {denom} candidates",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.25", fc=PLOS["white"], ec=PLOS["legend_edge"], lw=0.6),
        )
    else:
        ax.text(0.5, 0.5, "Candidate-recurrence data unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

    fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.21)
    save_figure(fig, out_base, args)


def build_panel_mapping() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "figure": "Fig 4",
            "panel": "A",
            "title": "Prioritization-stage reduction",
            "source_data_file": "Figure_4_panel_a_source_data.csv",
            "description": "Counts, percentages, and count provenance for each prioritization stage.",
        },
        {
            "figure": "Fig 4",
            "panel": "B",
            "title": "Multi-objective support scores",
            "source_data_file": "Figure_4_panel_b_source_data.csv",
            "description": "Median component scores with IQR across prioritization stages. Condition fidelity included when available.",
        },
        {
            "figure": "Fig 4",
            "panel": "C",
            "title": "Composite-score enrichment",
            "source_data_file": "Figure_4_panel_c_source_data.csv",
            "description": "Median composite/final score with IQR across prioritization stages.",
        },
        {
            "figure": "S13 Fig",
            "panel": "A",
            "title": "Ranking-scheme overlap",
            "source_data_file": "Supplementary_Figure_S13_panel_a_source_data.csv",
            "description": "Jaccard overlap matrix between alternative prioritization schemes; diagonal self-overlap is de-emphasized in the figure.",
        },
        {
            "figure": "S13 Fig",
            "panel": "B",
            "title": "Candidate recurrence across schemes",
            "source_data_file": "Supplementary_Figure_S13_panel_b_source_data.csv",
            "description": "Candidate counts and percentages by the number of prioritization schemes that recovered each candidate.",
        },
    ])


def readiness_score(checks: List[CheckResult]) -> Tuple[int, str]:
    fail = sum(c.status == "FAIL" for c in checks)
    warn = sum(c.status == "WARN" for c in checks)
    if fail:
        return max(60, 94 - fail * 6 - warn * 2), "FAIL"
    if warn:
        return max(90, 98 - warn), "WARN"
    return 100, "PASS"


def write_reports(
    dirs: OutputDirs,
    args: argparse.Namespace,
    inputs: InputPaths,
    discovery_records: List[DiscoveryRecord],
    checks: List[CheckResult],
    data: Dict[str, pd.DataFrame],
    files: Dict[str, str],
    component_metrics: List[Tuple[str, str]],
) -> None:
    score, status = readiness_score(checks)

    write_csv(pd.DataFrame([asdict(c) for c in checks]), dirs.reports / "step8_readiness_checks.csv")
    write_csv(pd.DataFrame([asdict(r) for r in discovery_records]), dirs.reports / "step8_discovery_records.csv")

    rejected = [r for r in discovery_records if r.decision in {"REJECT", "SKIP"}]
    write_csv(pd.DataFrame([asdict(r) for r in rejected]), dirs.reports / "step8_rejected_files.csv")

    report = []
    report.append("# OncoPep Step 8 readiness report\n\n")
    report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
    report.append(f"Script version: `{SCRIPT_VERSION}`\n\n")
    report.append(f"Overall status: **{status}**\n\n")
    report.append(f"Estimated readiness score: **{score}/100**\n\n")
    report.append("## Scientific role\n\n")
    report.append("Step 8 supports multi-objective prioritization and prioritization robustness. It does not assess anticancer activity, selectivity, toxicity, stability, receptor binding, or therapeutic efficacy.\n\n")

    report.append("## Figure logic improvements in this run\n\n")
    report.append("- Fig 4A uses log-scale prioritization-stage reduction and emphasizes the descriptor-plausible to shortlist/final compression.\n")
    report.append("- Fig 4B plots descriptive medians with IQR error bars. Condition fidelity is included only if a numeric condition-fidelity or condition-match column is available in all prioritization stages.\n")
    report.append("- Fig 4C reports composite-score enrichment with sample sizes in x-axis labels, not inside bars.\n")
    report.append("- S13A places the primary ranking first, uses complete scheme labels, and de-emphasizes diagonal self-overlap.\n")
    report.append("- S13B reports candidate recurrence as counts and percentages with an explicit denominator.\n\n")

    report.append("## Component metrics plotted in Fig 4B\n\n")
    for col, lab in component_metrics:
        report.append(f"- {lab}: `{col}`\n")
    if not any(col == "condition_fidelity_score" for col, _ in component_metrics):
        report.append("- Condition fidelity was not plotted because no numeric condition-fidelity column was available in all three stages.\n")

    report.append("\n## Selected input paths\n\n")
    for k, v in asdict(inputs).items():
        report.append(f"- {k}: `{v}`\n")

    report.append("\n## Validation checks\n\n")
    for c in checks:
        report.append(f"- **{c.status}** `{c.name}`: {c.detail}\n")

    report.append("\n## Output files\n\n")
    for k, v in files.items():
        report.append(f"- {k}: `{v}`\n")

    report.append("\n## Gate\n\n")
    if status == "PASS" and score >= 98:
        report.append("This Step 8 package passes the ≥98/100 readiness gate for figure review.\n")
    else:
        report.append("This Step 8 package is generated but contains WARN/FAIL checks. Review before manuscript writing.\n")

    (dirs.reports / "step8_readiness_report.md").write_text("".join(report), encoding="utf-8")

    manifest = {
        "script": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "step8_root": str(args.step8_root),
        "project_root": str(args.project_root),
        "inputs": {k: str(v) if v is not None else None for k, v in asdict(inputs).items()},
        "data_shapes": {k: list(v.shape) for k, v in data.items() if isinstance(v, pd.DataFrame)},
        "outputs": files,
        "checks": [asdict(c) for c in checks],
        "component_metrics_plotted": [{"column": c, "label": l} for c, l in component_metrics],
        "readiness_score": score,
        "readiness_status": status,
        "claim_boundary": "Computational prioritization and robustness only; no experimental anticancer validation.",
    }
    (dirs.reports / "step8_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    readme = f"""# OncoPep Step 8 output package

Script version: {SCRIPT_VERSION}

## Scientific role

Step 8 supports multi-objective prioritization and prioritization robustness.
It does not assess anticancer activity, selectivity, toxicity, stability, receptor binding, or therapeutic efficacy.

## Main outputs

- main_figure/Figure_4_prioritization_redesigned.png/pdf/tiff
- supplementary_figures/Supplementary_Figure_S13_prioritization_robustness_redesigned.png/pdf/tiff
- source_data/Figure_4_* and Supplementary_Figure_S13_*
- reports/step8_readiness_report.md
- reports/step8_manifest.json

Descriptor-distribution details are retained as source-data/report material only to avoid duplication with Fig 3.
"""
    (dirs.reports / "README_step8_outputs.txt").write_text(readme, encoding="utf-8")

    (dirs.code / "requirements_step8_minimal.txt").write_text(
        "\n".join([
            "python>=3.9",
            f"numpy=={np.__version__}",
            f"pandas=={pd.__version__}",
            f"matplotlib=={matplotlib.__version__}",
            "",
        ]),
        encoding="utf-8",
    )

    try:
        src = Path(__file__).resolve()
        if src.exists():
            shutil.copy2(src, dirs.code / SCRIPT_NAME)
    except Exception:
        pass


def validate_data(data: Dict[str, pd.DataFrame]) -> List[CheckResult]:
    checks: List[CheckResult] = []

    for key in ["passed_df", "shortlist_df", "final_panel_df"]:
        if key not in data or len(data[key]) == 0:
            checks.append(CheckResult(key, "FAIL", "Required table missing or empty."))
        else:
            checks.append(CheckResult(key, "PASS", f"{len(data[key]):,} rows."))

    component_metrics = available_component_metrics(data)
    checks.append(CheckResult(
        "component_scores",
        "PASS" if len(component_metrics) >= 1 else "FAIL",
        f"Available component metrics: {component_metrics}",
    ))

    for key in ["passed_df", "shortlist_df", "final_panel_df"]:
        if "final_score" in data[key].columns and data[key]["final_score"].notna().any():
            checks.append(CheckResult(f"{key}:final_score", "PASS", "final_score available."))
        else:
            checks.append(CheckResult(f"{key}:final_score", "FAIL", "final_score unavailable."))

    if "stability_df" in data and {"scheme_a", "scheme_b", "jaccard_overlap"}.issubset(data["stability_df"].columns):
        checks.append(CheckResult("stability_df", "PASS", "scheme overlap columns available."))
    else:
        checks.append(CheckResult("stability_df", "WARN", "scheme overlap data unavailable or incomplete."))

    if "recurrence_df" in data and "scheme_recurrence_n" in data["recurrence_df"].columns:
        checks.append(CheckResult("recurrence_df", "PASS", "candidate recurrence column available."))
    else:
        checks.append(CheckResult("recurrence_df", "WARN", "candidate recurrence data unavailable or incomplete."))

    return checks


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    dirs = ensure_dirs(args.step8_root)

    print(f"Hard-fix version: {SCRIPT_VERSION}")
    if args.unknown_args_ignored:
        print(f"Ignored unknown/Jupyter arguments: {args.unknown_args_ignored}")

    inputs, discovery_records = discover_inputs(args)
    data, checks = load_inputs(args, inputs)

    data, score_checks = ensure_score_columns(data)
    data, desc_checks = ensure_descriptor_columns(data)
    checks.extend(score_checks)
    checks.extend(desc_checks)

    validation_checks = validate_data(data)
    checks.extend(validation_checks)

    if any(c.status == "FAIL" for c in validation_checks):
        msg = [
            "\nERROR SUMMARY",
            "Step 8 validation failed after safe input loading.",
            "Selected inputs:",
        ]
        for k, v in asdict(inputs).items():
            msg.append(f"  {k}: {v}")
        msg.append("\nValidation checks:")
        for c in checks:
            msg.append(f"  [{c.status}] {c.name}: {c.detail}")
        msg.append("\nUse explicit --passed-file, --shortlist-file, --stability-file, and --recurrence-file paths.")
        raise RuntimeError("\n".join(msg))

    counts = stage_count_table(data, args)
    component_metrics = available_component_metrics(data)
    support_summary = summarize_scores(data, component_metrics, args)
    final_summary = final_score_summary(data, args)
    descriptor_summary = descriptor_distribution_summary(data)

    stability_mat, stability_long = stability_matrix_source(data.get("stability_df", pd.DataFrame()))
    recurrence_counts = recurrence_source(data.get("recurrence_df", pd.DataFrame()))

    files: Dict[str, str] = {}

    main_sources = {
        "Figure_4_panel_a_source_data": counts,
        "Figure_4_panel_b_source_data": support_summary,
        "Figure_4_panel_c_source_data": final_summary,
    }
    for name, df in main_sources.items():
        path = dirs.source_data / f"{name}.csv"
        write_csv(df, path)
        files[name] = str(path)

    fig4_all = combine_source_data({
        "Figure_4_panel_a": counts,
        "Figure_4_panel_b": support_summary,
        "Figure_4_panel_c": final_summary,
    })
    path = dirs.source_data / "Figure_4_source_data_all_panels.csv"
    write_csv(fig4_all, path)
    files["Figure_4_source_data_all_panels"] = str(path)

    supp_sources = {
        "Supplementary_Figure_S13_panel_a_source_data": stability_long,
        "Supplementary_Figure_S13_panel_b_source_data": recurrence_counts,
    }
    for name, df in supp_sources.items():
        path = dirs.source_data / f"{name}.csv"
        write_csv(df, path)
        files[name] = str(path)

    s13_all = combine_source_data({
        "Supplementary_Figure_S13_panel_a": stability_long,
        "Supplementary_Figure_S13_panel_b": recurrence_counts,
    })
    path = dirs.source_data / "Supplementary_Figure_S13_source_data_all_panels.csv"
    write_csv(s13_all, path)
    files["Supplementary_Figure_S13_source_data_all_panels"] = str(path)

    extra_sources = {
        "step8_prioritization_summary": fig4_all,
        "step8_selection_stage_counts": counts,
        "step8_candidate_support_summary": support_summary,
        "step8_scheme_overlap_summary": stability_long,
        "step8_candidate_recurrence_summary": recurrence_counts,
        "step8_descriptor_distribution_table": descriptor_summary,
        "step8_panel_source_data_mapping": build_panel_mapping(),
    }
    for name, df in extra_sources.items():
        path = dirs.source_data / f"{name}.csv"
        write_csv(df, path)
        files[name] = str(path)

    fig4_base = dirs.main_figure / "Figure_4_prioritization_redesigned"
    plot_figure4(counts, support_summary, final_summary, fig4_base, args)
    files["Figure_4_png"] = str(fig4_base.with_suffix(".png"))
    files["Figure_4_pdf"] = str(fig4_base.with_suffix(".pdf"))
    files["Figure_4_tiff"] = str(fig4_base.with_suffix(".tiff"))

    s13_base = dirs.supplementary_figures / "Supplementary_Figure_S13_prioritization_robustness_redesigned"
    plot_s13(stability_mat, recurrence_counts, s13_base, args)
    files["Supplementary_Figure_S13_png"] = str(s13_base.with_suffix(".png"))
    files["Supplementary_Figure_S13_pdf"] = str(s13_base.with_suffix(".pdf"))
    files["Supplementary_Figure_S13_tiff"] = str(s13_base.with_suffix(".tiff"))

    write_reports(dirs, args, inputs, discovery_records, checks, data, files, component_metrics)
    score, status = readiness_score(checks)

    print("\nOncoPep Step 8 package generated.")
    print(f"Root: {dirs.root}")
    print(f"Readiness status: {status}; estimated score: {score}/100")
    print(f"Main figure: {fig4_base.with_suffix('.png')}")
    print(f"Supplementary figure: {s13_base.with_suffix('.png')}")
    print(f"Readiness report: {dirs.reports / 'step8_readiness_report.md'}")
    if status != "PASS" or score < 98:
        print("WARNING: Package generated but contains WARN checks. Review readiness report before writing.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

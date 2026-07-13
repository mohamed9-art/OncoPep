"""Authoritative sequence descriptors used in the final OncoPep release."""

from __future__ import annotations


def candidate_charge_proxy(sequence: str) -> float:
    """Return the locked OncoPep sequence-composition charge proxy.

    Lysine and arginine contribute +1, histidine contributes +0.1,
    and aspartate and glutamate contribute -1. Terminal charges and
    residue-specific titration curves are not included.
    """
    sequence = sequence.strip().upper()

    return float(
        sequence.count("K")
        + sequence.count("R")
        + 0.1 * sequence.count("H")
        - sequence.count("D")
        - sequence.count("E")
    )

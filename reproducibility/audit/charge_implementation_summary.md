# OncoPep charge-implementation audit

## Final candidate-level charge proxy

The authoritative candidate-level charge proxy used in the final OncoPep candidate analyses is:

Q_proxy = n_K + n_R + 0.1 n_H - n_D - n_E

Lysine and arginine contribute +1, histidine contributes +0.1, and
aspartate and glutamate contribute -1. Terminal-group contributions
and residue-specific titration curves are not included.

The implementation was verified against:

- the 10,237-sequence descriptor-plausible pool;
- the 24-sequence shortlist;
- the final 12-candidate panel;
- the six literature-derived templates;
- the 18 literature-derived designed candidates.

For the authoritative `net_charge_proxy` fields, the calculated and
stored values matched exactly.

## Corpus input charge descriptor

The standardized corpus contains a precomputed field named
`net_charge_pH7`. Its original upstream calculation was not recovered
from the archived preprocessing code. It is therefore retained and
described as an input charge descriptor rather than as a newly
reproduced Henderson–Hasselbalch estimate.

## Superseded implementation

A legacy field named `charge_proxy` used a full +1 histidine
contribution. This implementation is superseded and is not the
authoritative candidate charge descriptor for the final OncoPep
analysis.

"""
TrialSense — Module 2 validation against real renal impairment data.

WHAT THIS MEASURES
------------------
For each drug we collected, from FDA labels and published studies, how much
drug exposure (AUC) actually rose in patients with impaired kidneys compared
with normal kidney function. Module 2 predicts the same number from nothing but
the drug's renally-excreted fraction (fe) and the patients' kidney function:

    KF        = min(1, GFR / 90)
    AUC_ratio = 1 / (fe * KF + (1 - fe))       (liver function held normal)

We then score the predictions the way PBPK models are scored in the literature:

  * fold error      max(pred/obs, obs/pred) — 1.0 is perfect
  * % within 1.25×, 1.5× and 2×   (2× is the usual acceptance bar for PBPK)
  * GMFE            geometric mean fold error across all observations
  * bias            geometric mean of pred/obs — below 1 means we under-predict
  * decision agreement: do prediction and reality fall in the same band
    (no change < 1.25×, adjust 1.25–2×, meaningful ≥ 2×)?
  * study-design agreement, per drug: does renal_study_plan() pick the design
    the real data justify? The data "need a full study" when any group shows a
    meaningful (≥ 2×) change, or a ≥ 1.25× change already at GFR ≥ 30 (mild or
    moderate impairment). Otherwise a reduced study would have been enough.
    This is the decision that spends or saves a study's worth of money, so a
    MISSED full study is the costly error; an unnecessary one is the cheap one.

Every observation in data/renal_validation.json carries a verbatim quote and
URL, so any number here can be traced back to its source.

HONEST LIMITS
-------------
  * Studies report kidney function as CrCl, eGFR or measured GFR, often as a
    range ("CrCl < 30"). We collapse each group to one representative GFR
    (see representative_gfr), and treat CrCl and GFR as interchangeable. Both
    add noise on top of the model's own error.
  * Clearance ratios are converted to AUC ratios (AUC = Dose / CL). Half-life-
    only observations are kept in the file but excluded from scoring.
  * Dialysis-only data are excluded: dialysis removes drug by a route the
    model does not represent. So are acute kidney injury in ICU patients and
    total-drug numbers distorted by protein-binding changes — each excluded
    record states its reason in the data file.
  * Some labels state plainly that kidney disease does NOT change exposure but
    give no number. Those are scored for decision agreement only (the observed
    band is "no change"), never for fold error.
  * This checks the ONE drug at a time exposure equation. It does not validate
    the interaction-severity weights or the 60/40 composite.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .drugs import DRUGS
from .pk import AUC_MEANINGFUL, AUC_NO_CHANGE, GFR_REFERENCE, renal_study_plan

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "renal_validation.json"

# Representative GFR (mL/min) for groups described only by name.
NAMED_GROUP_GFR = {
    "mild": 75.0,
    "moderate": 45.0,
    "severe": 22.0,
    "end-stage": 10.0,
    "esrd": 10.0,
    "kidney failure": 10.0,
}


def predict_auc_ratio(fe: float, gfr: float) -> float:
    """Module 2's exposure prediction for a renal-only change (liver normal)."""
    kf = min(1.0, gfr / GFR_REFERENCE)
    return 1.0 / max(0.05, fe * kf + (1.0 - fe))


def representative_gfr(obs: dict) -> float | None:
    """
    One GFR value (mL/min) standing in for a patient group.

      * explicit gfr_rep in the record wins (set during curation)
      * both bounds         → midpoint            ("CrCl 10-30"  → 20)
      * upper bound only    → 0.75 × upper        ("CrCl < 30"   → 22.5)
      * lower bound only    → lower + 15          ("CrCl > 50"   → 65)
      * name only           → NAMED_GROUP_GFR     ("severe"      → 22)
    Returns None for dialysis groups or anything unclassifiable.
    """
    if obs.get("gfr_rep") is not None:
        return float(obs["gfr_rep"])
    group = (obs.get("renal_group") or "").lower()
    if "dialysis" in group and "not on dialysis" not in group and "non-dialysis" not in group:
        return None
    lo, hi = obs.get("gfr_low"), obs.get("gfr_high")
    if lo is not None and hi is not None:
        return (float(lo) + float(hi)) / 2
    if hi is not None:
        return 0.75 * float(hi)
    if lo is not None:
        return float(lo) + 15
    for key, gfr in NAMED_GROUP_GFR.items():
        if key in group:
            return gfr
    return None


def decision_band(ratio: float) -> str:
    if ratio < AUC_NO_CHANGE:
        return "no change"
    if ratio < AUC_MEANINGFUL:
        return "adjust"
    return "meaningful"


@dataclass
class ScoredObservation:
    drug: str
    renal_group: str
    gfr: float
    observed: float
    predicted: float  # using the fe in drugs.py (what the app uses)
    predicted_label_fe: float | None  # using the fe reported by the source
    source_type: str
    url: str
    no_change_statement: bool = False  # label says "unchanged", no number

    @property
    def fold_error(self) -> float:
        return max(self.predicted / self.observed, self.observed / self.predicted)

    @property
    def band_agrees(self) -> bool:
        return decision_band(self.predicted) == decision_band(self.observed)


@dataclass
class ValidationResult:
    scored: list[ScoredObservation] = field(default_factory=list)
    n_drugs_curated: int = 0
    n_excluded: int = 0  # half-life only, dialysis, qualitative
    drugs_without_data: list[str] = field(default_factory=list)

    @property
    def numeric(self) -> list[ScoredObservation]:
        """Observations with a measured ratio (fold-error metrics use only these)."""
        return [s for s in self.scored if not s.no_change_statement]

    def _fraction(self, pred, pool=None) -> float:
        pool = self.numeric if pool is None else pool
        return sum(1 for s in pool if pred(s)) / len(pool) if pool else 0.0

    def design_calls(self) -> list[dict]:
        """Per-drug comparison of the recommended study design with the data."""
        calls = []
        for drug in sorted({s.drug for s in self.scored}):
            rows = [s for s in self.scored if s.drug == drug]
            needs_full = any(
                s.observed >= AUC_MEANINGFUL or (s.observed >= AUC_NO_CHANGE and s.gfr >= 30)
                for s in rows
            )
            predicted = renal_study_plan(DRUGS[drug]).design
            truth = "Full study" if needs_full else "Reduced study"
            calls.append({
                "drug": drug,
                "data_support": truth,
                "recommended": predicted,
                "outcome": ("correct" if predicted == truth
                            else "missed full study" if needs_full
                            else "unnecessary full study"),
            })
        return calls

    def summary(self) -> dict:
        if not self.numeric:
            return {"n_observations": 0}
        statements = [s for s in self.scored if s.no_change_statement]
        logs = [math.log10(s.predicted / s.observed) for s in self.numeric]
        with_label_fe = [s for s in self.numeric if s.predicted_label_fe is not None]
        label_logs = [math.log10(s.predicted_label_fe / s.observed) for s in with_label_fe]
        return {
            "n_observations": len(self.numeric),
            "n_drugs": len({s.drug for s in self.scored}),
            "n_no_change_statements": len(statements),
            "no_change_statements_matched": sum(1 for s in statements if s.band_agrees),
            "n_drugs_curated": self.n_drugs_curated,
            "n_excluded": self.n_excluded,
            "within_1_25x": self._fraction(lambda s: s.fold_error <= 1.25),
            "within_1_5x": self._fraction(lambda s: s.fold_error <= 1.5),
            "within_2x": self._fraction(lambda s: s.fold_error <= 2.0),
            "gmfe": 10 ** (sum(abs(x) for x in logs) / len(logs)),
            "bias": 10 ** (sum(logs) / len(logs)),
            "band_agreement": self._fraction(lambda s: s.band_agrees, self.scored),
            "meaningful_missed": sum(
                1 for s in self.numeric
                if decision_band(s.observed) == "meaningful"
                and decision_band(s.predicted) != "meaningful"
            ),
            "meaningful_observed": sum(
                1 for s in self.numeric if decision_band(s.observed) == "meaningful"
            ),
            "gmfe_with_label_fe": (
                10 ** (sum(abs(x) for x in label_logs) / len(label_logs)) if label_logs else None
            ),
            "n_with_label_fe": len(with_label_fe),
            "drugs_without_data": self.drugs_without_data,
            "design_correct": sum(1 for c in self.design_calls() if c["outcome"] == "correct"),
            "design_total": len(self.design_calls()),
            "design_missed_full": sum(
                1 for c in self.design_calls() if c["outcome"] == "missed full study"),
            "design_unnecessary_full": sum(
                1 for c in self.design_calls() if c["outcome"] == "unnecessary full study"),
        }


def load_records(path: Path = DATA_PATH) -> list[dict]:
    return json.loads(path.read_text())


def validate(path: Path = DATA_PATH) -> ValidationResult:
    records = load_records(path)
    result = ValidationResult(n_drugs_curated=len(records))
    for rec in records:
        drug = DRUGS.get(rec["drug"])
        if drug is None:
            continue
        label_fe = (rec.get("fe") or {}).get("value")
        usable = 0
        for obs in rec.get("observations", []):
            no_change = bool(obs.get("no_change"))
            ratio = 1.0 if no_change else obs.get("auc_ratio")
            gfr = representative_gfr(obs)
            if (obs.get("exclude_reason") or ratio is None or gfr is None
                    or obs.get("measure") == "half-life"):
                result.n_excluded += 1
                continue
            usable += 1
            result.scored.append(
                ScoredObservation(
                    drug=drug.name,
                    renal_group=obs.get("renal_group", ""),
                    gfr=round(gfr, 1),
                    observed=float(ratio),
                    predicted=round(predict_auc_ratio(drug.fe, gfr), 3),
                    predicted_label_fe=(
                        round(predict_auc_ratio(float(label_fe), gfr), 3)
                        if label_fe is not None else None
                    ),
                    source_type=obs.get("source_type", ""),
                    url=obs.get("url", ""),
                    no_change_statement=no_change,
                )
            )
        if not usable:
            result.drugs_without_data.append(drug.name)
    return result

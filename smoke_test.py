#!/usr/bin/env python3
"""
TrialSense — end-to-end smoke test.

Exercises every code path the Streamlit app uses, without needing a browser:
all five demo cases, custom SMILES input, uploaded-sequence handling, every
Plotly figure, and the edge cases that would otherwise only surface live
during a pitch.

    python smoke_test.py
"""

from __future__ import annotations

import traceback

PASS, FAIL = [], []


def check(name: str, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as exc:  # noqa: BLE001 — a smoke test wants every failure
        FAIL.append((name, exc))
        print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc()


def main() -> int:
    import pickle
    from pathlib import Path

    from trialsense import amr as amr_mod
    from trialsense import viz
    from trialsense.cases import DEMO_CASES
    from trialsense.ddi import (
        SCREENING_THRESHOLD,
        analyze_pair,
        knowledge_base_assess,
    )
    from trialsense.drugs import DRUGS, drug_from_smiles, validate_drug_db
    from trialsense.pk import PatientProfile, personalize
    from trialsense.report import build_report

    models = Path(__file__).parent / "models"
    print("=" * 72)
    print("TrialSense smoke test")
    print("=" * 72)

    # --- Structures ---------------------------------------------------------
    print("\n[Structures]")
    check("all 39 SMILES validate against reference MW",
          lambda: (_ for _ in ()).throw(AssertionError(validate_drug_db()))
          if validate_drug_db() else None)

    # --- Models load --------------------------------------------------------
    print("\n[Models]")
    ddi_model = amr_model = None
    if (models / "ddi_model.pkl").exists():
        with open(models / "ddi_model.pkl", "rb") as fh:
            ddi_model = pickle.load(fh)
        with open(models / "amr_model.pkl", "rb") as fh:
            amr_model = pickle.load(fh)
        check("cached models load", lambda: None)
    else:
        print("  SKIP  cached models absent — run `python train.py` first")
        return 1

    # --- Every demo case end to end -----------------------------------------
    print("\n[Demo cases]")
    for case in DEMO_CASES:
        def run(case=case):
            a, b = DRUGS[case.drug_a], DRUGS[case.drug_b]
            ddi = analyze_pair(a, b, ddi_model)
            assert ddi.ml_probability is not None, "no ML probability"
            assert 0.0 <= ddi.ml_probability <= 1.0
            pers = personalize(a, b, case.patient, ddi.headline_severity, ddi.kb_score)
            seq = amr_mod.demo_strain_sequence(case.strain)
            rep_amr = amr_mod.analyze_isolate(seq, amr_model, case.strain)
            rep = build_report(case.name, a, b, case.patient, ddi, pers, rep_amr)
            assert 0 <= rep.composite_score <= 100, rep.composite_score
            assert rep.findings, "no findings produced"
            assert rep.actions, "no actions produced"
            assert rep.headline()
            # every figure the app draws for this case
            viz.risk_gauge(rep.composite_score, rep.band, rep.color).to_json()
            viz.exposure_chart(pers.exposures).to_json()
            viz.resistance_chart(rep_amr.probabilities, amr_mod.LAST_RESORT).to_json()
            viz.screening_probability_chart(ddi.ml_probability, SCREENING_THRESHOLD).to_json()
            viz.molecule_svg(a.smiles)
            viz.molecule_svg(b.smiles)
            for f in rep.findings:
                _ = f.color, f.rank
        check(case.name.split("—")[0].strip() + f" ({case.drug_a}+{case.drug_b})", run)

    # --- Custom / novel SMILES path -----------------------------------------
    print("\n[Custom structures]")

    def novel_pair():
        a = drug_from_smiles("CC(=O)Nc1ccc(O)cc1", "Candidate TS-001")  # paracetamol
        b = drug_from_smiles("CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "Candidate TS-002")  # caffeine
        ddi = analyze_pair(a, b, ddi_model)
        assert not ddi.from_knowledge_base
        assert ddi.ml_probability is not None
        assert ddi.models_agree is None, "should not claim agreement without a KB entry"
        # novel + flagged must map to Moderate at most, never Severe
        assert ddi.headline_severity in (0, 2), ddi.headline_severity
        pers = personalize(a, b, PatientProfile(), ddi.headline_severity, ddi.kb_score)
        rep = build_report("novel", a, b, PatientProfile(), ddi, pers, None)
        assert not rep.amr_applicable
        assert rep.composite_score == rep.ddi_component
    check("novel SMILES pair scores without a knowledge-base entry", novel_pair)

    def novel_flagged_is_consistent():
        """
        A flagged novel pairing must not read Moderate on the interaction tab
        while the composite header reads 0/100 Low. Regression test: the app
        used to feed kb_score (always 0 for novel structures) into Module 2.
        """
        a = drug_from_smiles(DRUGS["Clarithromycin"].smiles, "TS-101")
        b = drug_from_smiles(DRUGS["Simvastatin"].smiles, "TS-102")
        ddi = analyze_pair(a, b, ddi_model)
        assert ddi.ml_flags_significant, "expected this structural pair to be flagged"
        assert ddi.headline_score > 1.4, ddi.headline_score
        p = PatientProfile()
        pers = personalize(a, b, p, ddi.headline_severity, ddi.headline_score)
        rep = build_report("novel", a, b, p, ddi, pers, None)
        assert pers.adjusted_severity >= 2, pers.adjusted_severity
        assert rep.composite_score > 30, rep.composite_score
        # …but a binary filter must never escalate a novel pair to Severe.
        assert ddi.headline_severity == 2, ddi.headline_severity
        assert pers.adjusted_severity <= 2 or pers.reasons, "severity 3 needs a reason"
    check("flagged novel pair scores consistently across modules", novel_flagged_is_consistent)

    def bad_smiles():
        a = drug_from_smiles("this-is-not-a-molecule", "Broken")
        ddi = analyze_pair(a, DRUGS["Warfarin"], ddi_model)
        assert ddi.ml_probability is None, "invalid SMILES must not yield a score"
        assert viz.molecule_svg("nonsense")  # renders a placeholder, does not crash
    check("invalid SMILES degrades gracefully", bad_smiles)

    # --- Sequence handling ---------------------------------------------------
    print("\n[Sequences]")

    def fasta_roundtrip():
        raw = ">test isolate\nACGTACGTNN\nGGCCTTAA\n"
        assert amr_mod.parse_fasta(raw) == "ACGTACGTNNGGCCTTAA"
        rep = amr_mod.analyze_isolate(
            amr_mod.demo_strain_sequence("K. pneumoniae (carbapenem-resistant)"),
            amr_model, "uploaded")
        assert "Carbapenems" in rep.resistant_classes(), rep.resistant_classes()
        assert rep.last_resort_hits()
    check("FASTA parsing and carbapenem detection", fasta_roundtrip)

    def tiny_sequence():
        rep = amr_mod.analyze_isolate("ACGT", amr_model, "tiny")
        assert rep.probabilities  # must not crash on a too-short input
        assert amr_mod.detect_genes("ACGT") == []
    check("very short sequence does not crash", tiny_sequence)

    def susceptible_strain():
        seq = amr_mod.demo_strain_sequence("E. coli ATCC 25922 (susceptible reference)")
        rep = amr_mod.analyze_isolate(seq, amr_model, "control")
        assert not rep.detected_genes, rep.detected_genes
        assert rep.risk_level()[0] == "Low", rep.risk_level()
    check("susceptible control strain reports no resistance", susceptible_strain)

    # --- Physiological edge cases -------------------------------------------
    print("\n[Edge cases]")

    def extreme_patient():
        p = PatientProfile(age=95, weight_kg=35, sex="Female", serum_creatinine=6.0,
                           alt=400, albumin=1.5, bilirubin=8.0, serum_potassium=6.5)
        assert p.creatinine_clearance() > 0
        assert 0.15 <= p.hepatic_function_factor() <= 1.0
        a, b = DRUGS["Tobramycin"], DRUGS["Furosemide"]
        sev, score, _ = knowledge_base_assess(a, b)
        pers = personalize(a, b, p, sev, score)
        assert all(e.exposure_ratio <= 20 for e in pers.exposures), "exposure blew up"
        assert pers.adjusted_severity == 3
    check("extreme organ failure stays bounded", extreme_patient)

    def healthy_patient():
        p = PatientProfile(age=25, weight_kg=80, serum_creatinine=0.7, alt=15,
                           albumin=4.8, bilirubin=0.4)
        a, b = DRUGS["Amoxicillin"], DRUGS["Metformin"]
        sev, score, _ = knowledge_base_assess(a, b)
        pers = personalize(a, b, p, sev, score)
        assert pers.adjusted_severity == 0, pers.adjusted_severity
        assert all(abs(e.exposure_ratio - 1.0) < 0.05 for e in pers.exposures)
    check("healthy profile on a clean pair stays green", healthy_patient)

    def symmetry():
        """A+B must equal B+A — the features are built to be order-invariant."""
        for a_name, b_name in [("Warfarin", "Fluconazole"), ("Digoxin", "Amiodarone")]:
            a, b = DRUGS[a_name], DRUGS[b_name]
            f = analyze_pair(a, b, ddi_model)
            r = analyze_pair(b, a, ddi_model)
            assert f.kb_severity == r.kb_severity
            assert abs((f.ml_probability or 0) - (r.ml_probability or 0)) < 1e-9, (
                f"{a_name}+{b_name}: {f.ml_probability} vs {r.ml_probability}")
    check("interaction scoring is order-invariant", symmetry)

    def self_pair():
        a = DRUGS["Warfarin"]
        ddi = analyze_pair(a, a, ddi_model)
        assert ddi.kb_severity >= 0  # must not crash on a drug paired with itself
    check("drug paired with itself does not crash", self_pair)

    def all_pairs_run():
        """Every pair in the knowledge base must produce a valid report."""
        import itertools
        names = sorted(DRUGS)
        p = PatientProfile(age=70, serum_creatinine=1.8, alt=90, albumin=3.0)
        for a_name, b_name in itertools.combinations(names, 2):
            a, b = DRUGS[a_name], DRUGS[b_name]
            sev, score, mechs = knowledge_base_assess(a, b)
            assert 0 <= sev <= 3
            pers = personalize(a, b, p, sev, score)
            assert 0 <= pers.adjusted_severity <= 3
            for m in mechs:
                assert m.text and m.consequence
    check("all 741 knowledge-base pairs produce valid output", all_pairs_run)

    # --- Report invariants ---------------------------------------------------
    print("\n[Report invariants]")

    def composite_bounds():
        p = PatientProfile(age=80, serum_creatinine=3.0, alt=200, albumin=2.5,
                           serum_potassium=5.8)
        a, b = DRUGS["Lisinopril"], DRUGS["Trimethoprim"]
        ddi = analyze_pair(a, b, ddi_model)
        pers = personalize(a, b, p, ddi.headline_severity, ddi.kb_score)
        seq = amr_mod.demo_strain_sequence("E. coli ST131 (ESBL)")
        rep = build_report("x", a, b, p, ddi, pers,
                           amr_mod.analyze_isolate(seq, amr_model, "s"))
        assert 0 <= rep.composite_score <= 100
        assert 0 <= rep.ddi_component <= 100
        assert 0 <= rep.amr_component <= 100
        assert rep.band in ("Low", "Moderate", "High", "Critical")
    check("composite score stays within 0-100", composite_bounds)

    print("\n" + "=" * 72)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    print("=" * 72)
    if FAIL:
        for name, exc in FAIL:
            print(f"  FAILED: {name}\n          {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

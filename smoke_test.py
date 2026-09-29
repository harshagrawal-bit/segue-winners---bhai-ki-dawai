#!/usr/bin/env python3
"""
TrialSense — end-to-end smoke test.

Exercises every code path the Streamlit app uses, without needing a browser:
all five demo cases, all six Module 3 features, custom SMILES input, every
Plotly figure, and the edge cases that would otherwise only surface live
during a pitch.

    python smoke_test.py
"""

from __future__ import annotations

import traceback

PASS, FAIL = [], []


def pytest_approx(value, tol=1e-6):
    """Compare a float for equality within tolerance, without pulling in pytest."""
    class _Approx:
        def __eq__(self, other):
            return abs(other - value) < tol
        def __repr__(self):
            return f"~{value}"
    return _Approx()


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
    from trialsense.pk import PatientProfile, personalize, renal_study_plan
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


    # --- Module 1 calibration against published severities -------------------
    # The only external validation Module 1 has. Everything else measures the
    # model against labels the knowledge base itself produced.
    print("\n[Module 1 · published-severity calibration]")
    from trialsense.known_pairs import KNOWN_PAIRS, evaluate as kp_eval

    _kp = kp_eval()

    def kp_no_under_calls():
        # The one that matters. An over-call costs a review; an under-call
        # tells a team a dangerous pairing is safe.
        under = [r for r in _kp["disagreements"] if r["ours"] < r["published"]]
        assert not under, "under-called: " + "; ".join(
            f"{r['pair']} published {r['published']} ours {r['ours']}" for r in under)
    check("no published-severe pair is under-called", kp_no_under_calls)

    def kp_direction():
        assert _kp["direction"] == _kp["n_pairs"], (
            f"{_kp['n_pairs'] - _kp['direction']} pair(s) disagree on whether the "
            "interaction is clinically significant")
    check("significant/not-significant agrees on all known pairs", kp_direction)

    def kp_within_one():
        assert _kp["within_one"] == _kp["n_pairs"], (
            f"{_kp['n_pairs'] - _kp['within_one']} pair(s) off by more than one level")
    check("every known pair is within one severity level", kp_within_one)

    def kp_exact_floor():
        # Guards against a change that quietly degrades calibration. Measured
        # at 21/27; the floor sits just below so normal variation does not fail
        # the build but a real regression does.
        assert _kp["exact"] >= 20, f"exact matches fell to {_kp['exact']}/{_kp['n_pairs']}"
    check("exact severity match holds at or above its measured floor", kp_exact_floor)

    def kp_antifolate_rule():
        # Regression test for the defect this calibration check found.
        from trialsense.ddi import knowledge_base_assess
        mtx_tmp, _, _ = knowledge_base_assess(DRUGS["Methotrexate"], DRUGS["Trimethoprim"])
        assert mtx_tmp == 3, f"methotrexate + trimethoprim should be Severe, got {mtx_tmp}"
        # …but co-trimoxazole is a deliberate combination and must not fire.
        _, _, mechs = knowledge_base_assess(DRUGS["Trimethoprim"], DRUGS["Sulfamethoxazole"])
        assert not any("folate" in m.text for m in mechs), (
            "the antifolate rule must not fire on a co-formulated pair")
    check("antifolate rule fires on methotrexate, spares co-trimoxazole",
          kp_antifolate_rule)

    def kp_coverage():
        assert len(KNOWN_PAIRS) >= 25, "reference set shrank"
        names = {d for kp in KNOWN_PAIRS for d in (kp.drug_a, kp.drug_b)}
        missing = [n for n in names if n not in DRUGS]
        assert not missing, f"reference pairs name unknown drugs: {missing}"
        assert any(kp.published == 0 for kp in KNOWN_PAIRS), "need negative controls"
        assert any(kp.published == 3 for kp in KNOWN_PAIRS), "need severe cases"
    check("calibration set is well-formed and spans the severity range", kp_coverage)

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


    # --- Module 3 features 1-6 -----------------------------------------------
    # Each feature produces its own independent report, so each is checked
    # independently here too.
    print("\n[Module 3 · Feature 1 — Resistance Runway]")
    from trialsense import surveillance as surv

    def runway_known():
        r = surv.build_runway_report("Klebsiella pneumoniae", "Carbapenems",
                                     "Meropenem", 2035)
        assert r.available, r.unavailable_reason
        assert r.verdict in ("VIABLE", "NARROWING", "CLOSING", "CLOSED")
        assert 0 <= r.launch_pct <= 100
        assert r.launch_lo <= r.launch_pct <= r.launch_hi
        assert r.headline() and r.plain_summary() and r.evidence_lines()
        # 2015 is a different denominator and must never enter the fit
        held = [p for p in r.points if not p.comparable]
        assert all(p.year == 2015 for p in held), "only 2015 may be held out"
        assert all(y >= 2016 for y in r.forecast.fitted), "fit used a 2015 point"
    check("runway report builds and excludes the 2015 denominator break", runway_known)

    def runway_unavailable():
        # M. tuberculosis is not an AMRSN surveillance organism. Reporting an
        # honest "no data" beats substituting a proxy.
        org = surv.organism_for_strain("M. tuberculosis (MDR-TB)")
        r = surv.build_runway_report(org, "Rifamycins", "Rifampicin", 2035)
        assert not r.available
        assert r.verdict == "UNKNOWN"
        assert r.unavailable_reason
    check("runway declines honestly when no surveillance series exists",
          runway_unavailable)

    def runway_monotone_sane():
        # A rising series must not project below its last observation.
        r = surv.build_runway_report("Acinetobacter baumannii", "Carbapenems",
                                     "Meropenem", 2040)
        assert r.launch_pct >= r.current_pct - 1.0
        assert r.launch_pct <= 100.0
    check("rising series projects upward and stays bounded", runway_monotone_sane)

    def forecast_backtest_runs():
        bt = surv.backtest_forecast(cutoff_year=2020)
        assert bt["available"]
        assert bt["n_predictions"] > 50
        assert 0 < bt["mae_pp"] < 25, bt["mae_pp"]
    check("forecast time-split backtest produces a sane error", forecast_backtest_runs)

    print("\n[Module 3 · Feature 2 — Rescue Strategy]")
    from trialsense import mechanisms as mech

    def rescue_metallo_vs_serine():
        # The distinction the whole feature exists for: identical class-level
        # verdict, opposite business decision.
        ndm = mech.build_rescue_report([("blaNDM-1", 0.96)], "Carbapenems", "X",
                                       "s", amr_mod.ANTIBIOTIC_CLASSES)
        kpc = mech.build_rescue_report([("blaKPC-2", 0.94)], "Carbapenems", "X",
                                       "s", amr_mod.ANTIBIOTIC_CLASSES)
        assert ndm.verdict == "DEAD END", ndm.verdict
        assert kpc.verdict == "SALVAGEABLE", kpc.verdict
        assert ndm.contrast_note() and kpc.contrast_note()
    check("metallo vs serine carbapenemase give opposite verdicts",
          rescue_metallo_vs_serine)

    def rescue_unscreened_class():
        # Metronidazole is Nitroimidazoles — not one of the twelve.
        r = mech.build_rescue_report([], "Nitroimidazoles", "Metronidazole", "s",
                                     amr_mod.ANTIBIOTIC_CLASSES)
        assert r.verdict == "NOT SCREENED"
        assert "not" in r.plain_summary().lower()
    check("unscreened antibiotic class reports NOT SCREENED, not 'clear'",
          rescue_unscreened_class)

    def rescue_worst_governs():
        # One rescuable and one unrescuable determinant must read DEAD END.
        r = mech.build_rescue_report(
            [("blaKPC-2", 0.9), ("blaNDM-1", 0.9)], "Carbapenems", "X", "s",
            amr_mod.ANTIBIOTIC_CLASSES)
        assert r.verdict == "DEAD END"
        assert r.blocking[0].rescue_viable == "no", "worst blocker must sort first"
    check("worst blocking determinant governs the rescue verdict",
          rescue_worst_governs)

    def all_genes_annotated():
        for name, g in amr_mod.RESISTANCE_GENES.items():
            assert g.mechanism_family != "unclassified", name
            assert g.rescue_viable in ("yes", "partial", "no", "unknown"), name
            assert g.mobility in ("plasmid", "integron", "transposon",
                                  "sccmec", "chromosomal"), name
            assert g.rescue_strategy and g.mobility_note, name
    check("all 18 genes carry mechanism and mobility annotations",
          all_genes_annotated)

    print("\n[Module 3 · Feature 3 — Portfolio Screen]")
    from trialsense import portfolio as port

    def portfolio_grid():
        cands = [
            port.Candidate("Meropenem", "Carbapenems"),
            port.Candidate("Linezolid", "Oxazolidinones"),
            port.Candidate("Levofloxacin", "Fluoroquinolones"),
        ]
        orgs = ["K. pneumoniae (carbapenem-resistant)",
                "E. coli ST131 (ESBL)",
                "S. aureus MRSA (hospital-acquired)"]

        def analyze(o):
            seq = amr_mod.demo_strain_sequence(o)
            return amr_mod.analyze_isolate(seq, amr_model, o)

        g = port.screen_portfolio(cands, orgs, analyze)
        assert len(g.cells) == 9
        assert len(g.scores) == 3
        assert g.n_combinations == 9
        assert g.elapsed_seconds > 0
        assert g.verdict in ("CLEAR LEADER", "CONTESTED", "NO VIABLE CANDIDATE")
        assert g.plain_summary() and g.recommendation()
        assert len(g.as_rows()) == 9
        # ranking must be sorted by breadth of viability
        v = [s.viable_against for s in g.scores]
        assert v == sorted(v, reverse=True), v
        viz.portfolio_heatmap(g).to_json()
    check("portfolio grid screens NxM and ranks by viability", portfolio_grid)

    def portfolio_unscreened_class():
        cands = [port.Candidate("Metronidazole", "Nitroimidazoles")]
        orgs = ["E. coli ST131 (ESBL)"]

        def analyze(o):
            return amr_mod.analyze_isolate(
                amr_mod.demo_strain_sequence(o), amr_model, o)

        g = port.screen_portfolio(cands, orgs, analyze)
        cell = g.cells[("Metronidazole", "E. coli ST131 (ESBL)")]
        assert not cell.screened
        assert cell.resistance is None
        assert cell.symbol == "—"
    check("portfolio marks unscreened classes rather than scoring them 0",
          portfolio_unscreened_class)

    print("\n[Module 3 · Feature 4 — Dark Genome]")
    from trialsense import novelty as nov

    def novelty_clean_strains():
        fp = 0
        for name in amr_mod.DEMO_STRAINS:
            seq = amr_mod.demo_strain_sequence(name)
            r = nov.build_novelty_report(seq, name, amr_model,
                                         len(amr_mod.detect_genes(seq)))
            if r.segments:
                fp += 1
        assert fp == 0, f"{fp} clean strains produced false positives"
    check("no false positives across all 12 clean demo strains",
          novelty_clean_strains)

    def novelty_detects_planted():
        import numpy as _np

        def foreign(gc, n, seed):
            r = _np.random.RandomState(seed)
            p = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
            mot = ["".join(r.choice(list("ACGT"), size=8, p=p)) for _ in range(14)]
            out = []
            while sum(len(x) for x in out) < n:
                out.append(mot[r.randint(len(mot))])
            return "".join(out)[:n]

        host = amr_mod.demo_strain_sequence("K. pneumoniae (carbapenem-resistant)")
        cut = len(host) // 2
        spiked = host[:cut] + foreign(0.22, 1200, 99) + host[cut:]
        r = nov.build_novelty_report(spiked, "spiked", amr_model, 0)
        assert r.segments, "planted unknown element was not detected"
        assert r.verdict in ("UNEXPLAINED SEQUENCE", "CAUTION", "UNRELIABLE")
        assert all(s.max_ref_similarity < nov.KNOWN_MATCH_THRESHOLD
                   for s in r.segments)
    check("planted unknown element is detected", novelty_detects_planted)

    def ood_badge_present():
        seq = amr_mod.demo_strain_sequence("E. coli ST131 (ESBL)")
        r = nov.build_novelty_report(seq, "x", amr_model, 0)
        assert r.ood_available, "retrain with train.py to store training stats"
        assert 0 <= r.ood_percentile <= 100
        assert r.ood_band in ("IN DISTRIBUTION", "BORDERLINE", "OUT OF DISTRIBUTION")
    check("out-of-distribution badge is available and bounded", ood_badge_present)

    def novelty_short_sequence():
        r = nov.build_novelty_report("ACGT" * 40, "tiny", None, 0)
        assert not r.scan_available
        assert r.verdict == "NOT ASSESSED"
        assert r.scan_unavailable_reason
    check("too-short sequence reports NOT ASSESSED rather than CLEAN",
          novelty_short_sequence)

    print("\n[Module 3 · Feature 5 — Value of Early Detection]")
    from trialsense import economics as econ

    def value_triggered():
        r = econ.build_value_report("Candidate-X", "K. pneu", 89.0, True)
        assert r.cost_without > r.cost_with
        assert r.expected_value_crore > 0
        assert 0 < r.probability_reaching_discovery <= 1
        assert r.verdict in ("HIGH VALUE", "MODERATE VALUE")
        assert r.disclaimer()
        viz.value_waterfall(r).to_json()
    check("value report computes a positive expected value when triggered",
          value_triggered)

    def value_not_triggered():
        # A clean screen must NOT claim a saving.
        r = econ.build_value_report("Candidate-Y", "E. coli", 12.0, False)
        assert r.expected_value_crore == 0.0
        assert r.verdict == "CONFIRMATORY"
        assert "no avoided cost" in r.plain_summary().lower() \
            or "not" in r.plain_summary().lower()
    check("clean screen claims no saving", value_not_triggered)

    print("\n[Module 3 · Feature 6 — Spread Velocity]")

    def mobility_plasmid_vs_chromosome():
        plasmid = mech.build_mobility_report([("blaNDM-1", 0.96)], "s")
        chrom = mech.build_mobility_report([("rpoB_S450L", 0.95)], "s")
        assert plasmid.multiplier > chrom.multiplier
        assert plasmid.verdict == "HIGHLY MOBILE"
        assert chrom.verdict == "STATIC"
        assert chrom.multiplier == 1.0
        viz.mobility_chart(plasmid).to_json()
    check("plasmid determinants score faster spread than chromosomal ones",
          mobility_plasmid_vs_chromosome)

    def mobility_fastest_governs():
        # One plasmid gene among chromosomal ones still spreads at plasmid speed.
        m = mech.build_mobility_report(
            [("gyrA_S83L", 0.9), ("rpoB_S450L", 0.9), ("mcr-1", 0.9)], "s")
        assert m.multiplier == 2.5, m.multiplier
        assert m.last_resort_mobile(), "mcr-1 defeats a last-resort class"
    check("fastest determinant governs the spread multiplier",
          mobility_fastest_governs)

    def mobility_empty():
        m = mech.build_mobility_report([], "clean")
        assert m.verdict == "NONE DETECTED"
        assert m.multiplier == 1.0
        viz.mobility_chart(m).to_json()
    check("no determinants gives a neutral multiplier", mobility_empty)


    print("\n[Module 3 · Feature 3 — ICMR cross-check]")

    def crosscheck_runs():
        strains = ["K. pneumoniae (carbapenem-resistant)", "E. coli ST131 (ESBL)"]
        reports = {
            s: amr_mod.analyze_isolate(amr_mod.demo_strain_sequence(s), amr_model, s)
            for s in strains
        }
        cc = surv.cross_check_predictions(
            reports,
            amr_mod.ANTIBIOTIC_CLASSES,
            {s: amr_mod.DEMO_STRAINS[s].genes for s in strains},
            {g: gg.confers_resistance_to for g, gg in amr_mod.RESISTANCE_GENES.items()},
        )
        assert cc.rows, "no comparison rows produced"
        assert cc.compared, "nothing had an ICMR reference"
        assert cc.verdict in (
            "CONSISTENT", "MOSTLY CONSISTENT", "DIVERGENT", "NO REFERENCE DATA")
        assert cc.reference_year == 2024, cc.reference_year
        assert cc.headline() and cc.plain_summary() and cc.caveat()
        for r in cc.rows:
            assert r.status in ("ALIGNED", "AS EXPECTED", "CONCERN", "NO REFERENCE")
            assert r.explain()
            assert 0 <= r.model_pct <= 100
    check("ICMR cross-check produces comparable rows", crosscheck_runs)

    def crosscheck_direction_logic():
        # A strain carrying a determinant reading ABOVE national is expected,
        # not a failure. Reading BELOW is the real concern.
        above = surv.CrossCheckRow(
            organism="X", strain_name="s", antibiotic_class="Carbapenems",
            model_pct=90.0, real_pct=60.0, real_year=2024,
            carries_determinant=True, determinants=["blaNDM-1"])
        below = surv.CrossCheckRow(
            organism="X", strain_name="s", antibiotic_class="Carbapenems",
            model_pct=20.0, real_pct=60.0, real_year=2024,
            carries_determinant=True, determinants=["blaNDM-1"])
        clean_high = surv.CrossCheckRow(
            organism="X", strain_name="s", antibiotic_class="Carbapenems",
            model_pct=90.0, real_pct=20.0, real_year=2024,
            carries_determinant=False)
        clean_low = surv.CrossCheckRow(
            organism="X", strain_name="s", antibiotic_class="Carbapenems",
            model_pct=5.0, real_pct=60.0, real_year=2024,
            carries_determinant=False)
        assert above.status == "AS EXPECTED", above.status
        assert below.status == "CONCERN", below.status       # under-call
        assert clean_high.status == "CONCERN", clean_high.status  # over-call
        assert clean_low.status == "AS EXPECTED", clean_low.status
    check("cross-check judges against expected direction, not raw gap",
          crosscheck_direction_logic)

    def crosscheck_alignment_band():
        tight = surv.CrossCheckRow(
            organism="X", strain_name="s", antibiotic_class="Carbapenems",
            model_pct=66.0, real_pct=60.0, real_year=2024,
            carries_determinant=True, determinants=["blaNDM-1"])
        assert tight.status == "ALIGNED", tight.status
        assert abs(tight.delta - 6.0) < 1e-9
    check("small gaps count as aligned regardless of direction",
          crosscheck_alignment_band)

    def crosscheck_large_expected_flagged():
        # Carrying the gene explains a higher reading, not an arbitrary one.
        huge = surv.CrossCheckRow(
            organism="X", strain_name="s", antibiotic_class="Aminoglycosides",
            model_pct=88.0, real_pct=28.0, real_year=2024,
            carries_determinant=True, determinants=["aac(6')-Ib"])
        assert huge.status == "AS EXPECTED"
        assert huge.large_for_direction, "wide expected gap should still be flagged"
        assert "wide gap" in huge.explain()
    check("wide expected gaps are still surfaced for review",
          crosscheck_large_expected_flagged)

    def crosscheck_no_reference_honest():
        # M. tuberculosis is not an AMRSN organism — must not be silently scored.
        row = surv.CrossCheckRow(
            organism="X", strain_name="s", antibiotic_class="Rifamycins",
            model_pct=95.0, real_pct=None, real_year=None)
        assert row.status == "NO REFERENCE"
        assert row.delta is None
        assert "no national reference" in row.explain().lower() \
            or "no " in row.explain().lower()
    check("missing reference data is reported, not scored as agreement",
          crosscheck_no_reference_honest)

    def crosscheck_attaches_to_grid():
        from trialsense import portfolio as port2
        cands = [port2.Candidate("Meropenem", "Carbapenems")]
        orgs = ["K. pneumoniae (carbapenem-resistant)"]

        def analyze(o):
            return amr_mod.analyze_isolate(
                amr_mod.demo_strain_sequence(o), amr_model, o)

        g = port2.screen_portfolio(cands, orgs, analyze)
        assert g.cross_check is None, "should not run until attached"
        port2.attach_cross_check(
            g,
            {o: amr_mod.DEMO_STRAINS[o].genes for o in orgs},
            {n: gg.confers_resistance_to
             for n, gg in amr_mod.RESISTANCE_GENES.items()},
        )
        assert g.cross_check is not None
        assert g.cross_check.compared
    check("portfolio grid carries its ICMR cross-check", crosscheck_attaches_to_grid)


    print("\n[Module 3 · per-drug resolution]")
    from trialsense import perdrug as pd_mod

    def perdrug_gentamicin_spared():
        # The headline fix: aac(6')-Ib must NOT write off gentamicin.
        r = pd_mod.resolve_gene("aac(6')-Ib", "Aminoglycosides")
        assert r.resolved
        assert "amikacin" in r.defeated
        assert "gentamicin" not in r.defeated, "aac(6')-Ib does not modify gentamicin"
        assert "gentamicin" in r.spared, "gentamicin must be reported as spared"
        assert "plazomicin" in r.engineered_escape
    check("aac(6')-Ib spares gentamicin", perdrug_gentamicin_spared)

    def perdrug_arma_opposite():
        # armA has the opposite profile — it DOES take gentamicin.
        r = pd_mod.resolve_gene("armA", "Aminoglycosides")
        assert "gentamicin" in r.defeated, "armA defeats gentamicin"
        assert "gentamicin" not in r.spared
    check("armA defeats gentamicin, unlike aac(6')-Ib", perdrug_arma_opposite)

    def perdrug_no_false_sparing():
        # Absence from a poorly curated database must never become a claim.
        # 16S methyltransferases DO defeat plazomicin; CARD just omits it.
        r = pd_mod.resolve_gene("armA", "Aminoglycosides")
        assert "plazomicin" not in r.spared, (
            "must not infer plazomicin sparing from a database gap")
        v = pd_mod.class_verdict([r], "Aminoglycosides")
        assert "plazomicin" not in v, v
    check("no sparing claim is inferred from a database gap",
          perdrug_no_false_sparing)

    def perdrug_class_only_honest():
        # CARD has only class-level links for KPC — we must not invent detail.
        r = pd_mod.resolve_gene("blaKPC-2", "Carbapenems")
        assert not r.resolved
        assert not r.spared
        assert "do not enumerate" in r.summary()
    check("class-level-only genes report no invented per-drug detail",
          perdrug_class_only_honest)

    def perdrug_verdict_combines():
        # Two genes together must cover more than either alone.
        aac = pd_mod.resolve_gene("aac(6')-Ib", "Aminoglycosides")
        arma = pd_mod.resolve_gene("armA", "Aminoglycosides")
        one = pd_mod.class_verdict([aac], "Aminoglycosides")
        both = pd_mod.class_verdict([aac, arma], "Aminoglycosides")
        assert "gentamicin" in one, "gentamicin survives aac(6')-Ib alone"
        assert "exhausted" in both or "gentamicin" not in both.split("but")[-1]
    check("combined determinants narrow the surviving options",
          perdrug_verdict_combines)

    print("\n[Module 3 · real-data ingestion]")

    def computational_predictions_never_become_labels():
        # BV-BRC's genome_amr table mixes bench measurements with other
        # groups' ML predictions, flagged in its `evidence` column. Scoring
        # our model against their model and calling it accuracy would be
        # meaningless, so the aggregator must drop every non-laboratory row.
        from trialsense import ingest
        rows = [
            {"genome_id": "1.1", "genome_name": "real", "taxon_id": 562,
             "antibiotic": "ciprofloxacin", "resistant_phenotype": "Resistant",
             "evidence": "Laboratory Method"},
            {"genome_id": "1.1", "genome_name": "real", "taxon_id": 562,
             "antibiotic": "gentamicin", "resistant_phenotype": "Resistant",
             "evidence": "Computational Method"},
        ]
        isolates, stats = ingest.aggregate_to_classes(rows)
        labels = isolates["1.1"].class_labels
        assert labels == {"Fluoroquinolones": 1}, labels
        assert "Aminoglycosides" not in labels, "a prediction became a label"
        assert stats["dropped_non_lab_evidence"] == 1
    check("computational predictions are never used as laboratory labels",
          computational_predictions_never_become_labels)

    def intermediate_results_are_dropped_not_coerced():
        from trialsense import ingest
        rows = [{"genome_id": "2.1", "genome_name": "x", "taxon_id": 573,
                 "antibiotic": "meropenem", "resistant_phenotype": "Intermediate",
                 "evidence": "Laboratory Method"}]
        isolates, stats = ingest.aggregate_to_classes(rows)
        assert not isolates, "an Intermediate result was coerced into a label"
        assert stats["dropped_intermediate"] == 1
    check("CLSI Intermediate is dropped rather than forced to R or S",
          intermediate_results_are_dropped_not_coerced)

    def ingested_labels_are_bench_measurements():
        # Guards the saved label set, if ingestion has been run.
        import json
        from pathlib import Path
        p = Path("data/real/lab_ast_labels.json")
        if not p.exists():
            return  # ingestion not run in this checkout; nothing to guard
        data = json.loads(p.read_text())
        assert data, "label file exists but is empty"
        for gid, rec in data.items():
            assert rec["class_labels"], f"{gid} carries no labels"
            for ev in rec["evidence"]:
                assert ev["class"] in amr_mod.ANTIBIOTIC_CLASSES, ev
    check("saved real-isolate labels are class-valid bench results",
          ingested_labels_are_bench_measurements)

    print("\n[Module 3 · point-mutation alleles]")

    def wild_type_locus_is_not_called_resistant():
        # THE BUG: gyrA exists in every E. coli, and our reference is the
        # wild-type K-12 gene. Detection matched it at ~0.999, and the
        # mechanism floor then raised Fluoroquinolones to 0.89 on a fully
        # susceptible isolate. Presence of the locus must prove nothing.
        wt = amr_mod._load_reference("gyrA_S83L")
        state, _ = amr_mod.read_allele(wt, "gyrA_S83L")
        assert state == amr_mod.ALLELE_WILD_TYPE, state
        out = amr_mod.apply_mechanism_floor(
            {"Fluoroquinolones": 0.02}, [("gyrA_S83L", 0.99)], sequence=wt)
        assert out["Fluoroquinolones"] == 0.02, out
    check("wild-type gyrA locus does not trigger the resistance floor",
          wild_type_locus_is_not_called_resistant)

    def resistant_allele_still_triggers_the_floor():
        # The fix must not cost us the true positive it exists to keep.
        wt = amr_mod._load_reference("gyrA_S83L")
        i = (83 - 1) * 3
        mutant = wt[:i] + "TTG" + wt[i + 3:]
        state, _ = amr_mod.read_allele(mutant, "gyrA_S83L")
        assert state == amr_mod.ALLELE_RESISTANT, state
        out = amr_mod.apply_mechanism_floor(
            {"Fluoroquinolones": 0.02}, [("gyrA_S83L", 0.99)], sequence=mutant)
        # The floor is the gene's own confirmed-allele value, scaled by
        # detection confidence — not a flat constant.
        expected = amr_mod.mechanism_floor_for("Fluoroquinolones", "gyrA_S83L") * 0.99
        assert abs(out["Fluoroquinolones"] - expected) < 1e-6, out
        assert out["Fluoroquinolones"] > 0.02, "the floor did not fire at all"
    check("S83L resistant allele does trigger the floor",
          resistant_allele_still_triggers_the_floor)

    def allele_is_read_on_either_strand():
        # Assembly contigs come in either orientation.
        wt = amr_mod._load_reference("gyrA_S83L")
        i = (83 - 1) * 3
        mutant = wt[:i] + "TTG" + wt[i + 3:]
        rc = mutant.translate(str.maketrans("ACGT", "TGCA"))[::-1]
        assert amr_mod.read_allele(rc, "gyrA_S83L")[0] == amr_mod.ALLELE_RESISTANT
    check("resistant allele is found on the reverse strand too",
          allele_is_read_on_either_strand)

    def unverifiable_allele_asserts_nothing():
        # No sequence to check, and an unrelated sequence, must both decline to
        # assert resistance rather than guessing in either direction.
        out = amr_mod.apply_mechanism_floor(
            {"Fluoroquinolones": 0.02}, [("gyrA_S83L", 0.99)])
        assert out["Fluoroquinolones"] == 0.02, out
        state, _ = amr_mod.read_allele("ACGT" * 300, "gyrA_S83L")
        assert state == amr_mod.ALLELE_UNDETERMINED, state
    check("an allele we cannot read is never asserted as resistant",
          unverifiable_allele_asserts_nothing)

    def ancestral_shv_does_not_claim_cephalosporins():
        # SHV-1 is a narrow-spectrum penicillinase, and it is what most
        # Klebsiella carry chromosomally. Mapping blaSHV to Cephalosporins at
        # class level called every one of them cephalosporin-resistant from
        # the locus alone. Only the Gly238Ser variant earns that claim.
        shv1 = amr_mod._load_reference("blaSHV")
        classes, _ = amr_mod.classes_conferred("blaSHV", shv1)
        assert "Penicillins" in classes, classes
        assert "Cephalosporins" not in classes, classes
        out = amr_mod.apply_mechanism_floor(
            {"Penicillins": 0.1, "Cephalosporins": 0.1},
            [("blaSHV", 0.99)], sequence=shv1)
        assert out["Cephalosporins"] == 0.1, out
        assert out["Penicillins"] == pytest_approx(
            amr_mod.mechanism_floor_for("Penicillins", "blaSHV") * 0.99), out
    check("ancestral SHV-1 does not assert cephalosporin resistance",
          ancestral_shv_does_not_claim_cephalosporins)

    def esbl_shv_does_claim_cephalosporins():
        shv1 = amr_mod._load_reference("blaSHV")
        i = 233 * 3
        esbl = shv1[:i] + "AGC" + shv1[i + 3:]
        classes, _ = amr_mod.classes_conferred("blaSHV", esbl)
        assert "Cephalosporins" in classes, classes
        out = amr_mod.apply_mechanism_floor(
            {"Cephalosporins": 0.1}, [("blaSHV", 0.99)], sequence=esbl)
        assert out["Cephalosporins"] == pytest_approx(
            amr_mod.mechanism_floor_for("Cephalosporins", "blaSHV") * 0.99), out
    check("SHV Gly238Ser does assert cephalosporin resistance",
          esbl_shv_does_claim_cephalosporins)

    print("\n[Module 3 · cross-cutting]")

    def unscreened_class_not_scored_as_zero():
        # The regression this whole bug-fix exists for: metronidazole's class
        # is not screened, and must not read as "0% resistance, susceptible".
        a, b = DRUGS["Metronidazole"], DRUGS["Warfarin"]
        p = PatientProfile()
        ddi = analyze_pair(a, b, ddi_model)
        pers = personalize(a, b, p, ddi.headline_severity, ddi.headline_score)
        seq = amr_mod.demo_strain_sequence("K. pneumoniae (carbapenem-resistant)")
        rep = build_report("x", a, b, p, ddi, pers,
                           amr_mod.analyze_isolate(seq, amr_model, "s"))
        assert not rep.amr_class_screened
        assert rep.amr_component == 0.0
        # composite_score is rounded for display; ddi_component is not
        assert abs(rep.composite_score - rep.ddi_component) < 0.1
        titles = " ".join(f.title for f in rep.findings)
        assert "not screened" in titles.lower()
        assert "remains susceptible" not in titles.lower()
    check("unscreened antibiotic class is never reported as susceptible",
          unscreened_class_not_scored_as_zero)

    def dnabert_backend_degrades():
        # The optional path must report its absence, never raise.
        from trialsense import dnabert as bert
        s = bert.check_backend()
        assert isinstance(s.available, bool)
        assert s.message()
        if not s.available:
            assert s.reason
    check("DNABERT-2 backend probe never raises", dnabert_backend_degrades)

    # --- Module 2: kidney function and trial design ---------------------------
    print("\n[Module 2 · kidney function and study plan]")

    def ckd_epi_reference():
        # Published CKD-EPI 2021 reference: 50-year-old man, SCr 1.0 → eGFR 92.
        p = PatientProfile(age=50, sex="Male", serum_creatinine=1.0)
        assert round(p.egfr()) == 92, p.egfr()
        # Higher creatinine must always mean lower eGFR.
        worse = PatientProfile(age=50, sex="Male", serum_creatinine=2.0)
        assert worse.egfr() < p.egfr()
        assert worse.renal_category() == "Moderate", worse.renal_category()
    check("CKD-EPI 2021 matches the published reference value", ckd_epi_reference)

    def absolute_egfr():
        # At a body surface area of exactly 1.73 m², absolute = indexed eGFR.
        p = PatientProfile(weight_kg=70, height_cm=175)
        scale = p.body_surface_area() / 1.73
        assert abs(p.absolute_egfr() - p.egfr() * scale) < 0.2
        small = PatientProfile(weight_kg=45, height_cm=150)
        assert small.absolute_egfr() < small.egfr(), "small body must de-index downward"
    check("absolute eGFR scales with body surface area", absolute_egfr)

    def healthy_adult_unchanged():
        # A healthy average adult must not show inflated exposure for a
        # renally cleared drug (regression: the old CrCl/120 factor did).
        pers = personalize(DRUGS["Lisinopril"], DRUGS["Amoxicillin"], PatientProfile(), 0, 0.0)
        assert all(abs(e.exposure_ratio - 1.0) < 0.02 for e in pers.exposures), pers.exposures
    check("healthy average adult shows unchanged renal exposure", healthy_adult_unchanged)

    def study_plan():
        lis = renal_study_plan(DRUGS["Lisinopril"])
        assert lis.design == "Full study"
        assert lis.severe.exposure_ratio >= 2.0
        sim = renal_study_plan(DRUGS["Simvastatin"])
        assert sim.design == "Reduced study"
        assert sim.model_supported_candidate
        dig = renal_study_plan(DRUGS["Digoxin"])  # narrow margin, ≥2× in severe
        assert dig.severe.enrolment.startswith("Exclude"), dig.severe.enrolment
        # Exposure must rise monotonically as kidney function falls.
        for name in DRUGS:
            r = [c.exposure_ratio for c in renal_study_plan(DRUGS[name]).categories]
            assert r == sorted(r), (name, r)
    check("renal study plan picks full vs reduced designs sensibly", study_plan)

    def control_case_stays_clean():
        # The healthy control case must not pick up renal study actions.
        case = DEMO_CASES[-1]
        a, b = DRUGS[case.drug_a], DRUGS[case.drug_b]
        ddi = analyze_pair(a, b, ddi_model)
        pers = personalize(a, b, case.patient, ddi.headline_severity, ddi.kb_score)
        rep = build_report(case.name, a, b, case.patient, ddi, pers, None)
        assert not any("renal impairment study" in x for x in rep.actions), rep.actions
    check("healthy control case gets no renal study actions", control_case_stays_clean)

    def real_data_validation():
        from trialsense.pk_validation import validate
        res = validate()
        s = res.summary()
        assert s["n_observations"] >= 20, s["n_observations"]
        # Every scored number must be traceable to a source.
        assert all(o.url for o in res.scored), "observation without a source URL"
        # The costly error: recommending a reduced study when real data show
        # the kidney matters. Guard it so a model change cannot silently
        # reintroduce it.
        assert s["design_missed_full"] == 0, res.design_calls()
    check("renal validation: no full study missed on real data", real_data_validation)

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

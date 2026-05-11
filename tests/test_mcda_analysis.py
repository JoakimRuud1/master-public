"""Smoketester for MCDA-modulen."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.mcda_analysis import (
    DEFAULT_WEIGHTS,
    MCDAConfig,
    build_normalised_matrix,
    compute_mcda_table,
    normalise_score,
    parse_weights,
    perturb_weights,
    run_mcda,
    sensitivity_analysis,
    weighted_score,
)


def _toy_aggregates() -> pd.DataFrame:
    """Lager en miniatyrversjon av strategy_aggregates.csv."""
    rows = [
        {
            "strategy_id": "A",
            "mean_primary_score": 5.0,
            "mean_citation": 5.0,
            "mean_accurate": 5.0,
            "mean_thorough": 5.0,
            "mean_useful": 5.0,
            "mean_organized": 5.0,
            "mean_comprehensible": 5.0,
            "mean_succinct": 5.0,
            "mean_synthesized_when_applicable": 5.0,
            "introduced_stigmatizing_language_rate": 0.0,
        },
        {
            "strategy_id": "B",
            "mean_primary_score": 1.0,
            "mean_citation": 1.0,
            "mean_accurate": 1.0,
            "mean_thorough": 1.0,
            "mean_useful": 1.0,
            "mean_organized": 1.0,
            "mean_comprehensible": 1.0,
            "mean_succinct": 1.0,
            "mean_synthesized_when_applicable": 1.0,
            "introduced_stigmatizing_language_rate": 0.0,
        },
        {
            "strategy_id": "C",
            "mean_primary_score": 4.3,
            "mean_citation": 4.0,
            "mean_accurate": 5.0,
            "mean_thorough": 3.0,
            "mean_useful": 4.0,
            "mean_organized": 5.0,
            "mean_comprehensible": 5.0,
            "mean_succinct": 5.0,
            "mean_synthesized_when_applicable": 4.0,
            "introduced_stigmatizing_language_rate": 0.5,
        },
    ]
    return pd.DataFrame(rows)


def test_normalise_endpoints():
    assert normalise_score(1.0) == 0.0
    assert normalise_score(5.0) == 1.0
    assert normalise_score(3.0) == 0.5


def test_perfect_strategy_gets_score_one():
    df = _toy_aggregates()
    cfg = MCDAConfig()
    table = compute_mcda_table(df, cfg)
    perfect = table[table["strategy_id"] == "A"].iloc[0]
    worst = table[table["strategy_id"] == "B"].iloc[0]
    assert perfect["mcda_score"] == pytest.approx(1.0)
    assert worst["mcda_score"] == pytest.approx(0.0)


def test_ranking_uses_weighted_sum():
    df = _toy_aggregates()
    cfg = MCDAConfig()
    table = compute_mcda_table(df, cfg)
    assert list(table["strategy_id"]) == ["A", "C", "B"]


def test_perturb_keeps_sum_one():
    base = dict(DEFAULT_WEIGHTS)
    perturbed = perturb_weights(base, "accurate", 1.25)
    assert sum(perturbed.values()) == pytest.approx(1.0)
    # Den perturberte vekten skal være større enn den normaliserte basen
    base_norm = {k: v / sum(base.values()) for k, v in base.items()}
    assert perturbed["accurate"] > base_norm["accurate"]


def test_sensitivity_table_shape():
    df = _toy_aggregates()
    cfg = MCDAConfig()
    sens = sensitivity_analysis(df, cfg)
    # En +/- per dimensjon
    assert len(sens) == 2 * len(cfg.weights)
    assert {"perturbed_weight", "direction", "top1_changed"}.issubset(sens.columns)


def test_missing_synthesis_dimension_renormalises():
    df = _toy_aggregates().copy()
    df.loc[df["strategy_id"] == "A", "mean_synthesized_when_applicable"] = float("nan")
    cfg = MCDAConfig()
    norm = build_normalised_matrix(df, list(cfg.weights.keys()))
    score, used = weighted_score(norm.loc["A"], cfg.normalised_weights())
    # Skal fortsatt gi 1.0 fordi alle øvrige dimensjoner er 5/5
    assert score == pytest.approx(1.0)
    assert "synthesized" not in used
    assert sum(used.values()) == pytest.approx(1.0)


def test_run_mcda_writes_files(tmp_path: Path):
    df = _toy_aggregates()
    csv_path = tmp_path / "strategy_aggregates.csv"
    df.to_csv(csv_path, index=False)
    results = run_mcda(tmp_path, do_sensitivity=True)
    assert (tmp_path / "mcda_results.csv").exists()
    assert (tmp_path / "mcda_vs_primary_score.csv").exists()
    assert (tmp_path / "mcda_sensitivity.csv").exists()
    assert (tmp_path / "mcda_weights.json").exists()
    payload = json.loads((tmp_path / "mcda_weights.json").read_text(encoding="utf-8"))
    assert payload["weights_normalised"]["accurate"] == pytest.approx(
        DEFAULT_WEIGHTS["accurate"] / sum(DEFAULT_WEIGHTS.values())
    )
    assert "mcda_table" in results


def test_parse_weights_inline():
    weights = parse_weights("accurate=0.5,thorough=0.5")
    assert weights == {"accurate": 0.5, "thorough": 0.5}


def test_parse_weights_from_file(tmp_path: Path):
    p = tmp_path / "w.json"
    p.write_text(json.dumps({"accurate": 0.3, "thorough": 0.7}))
    weights = parse_weights(str(p))
    assert weights == {"accurate": 0.3, "thorough": 0.7}


def test_sensitivity_plots_smoke(tmp_path: Path):
    """Sjekker at alle tre figurfilene blir skrevet og er ikke-tomme."""
    from src.mcda_plots import make_all_sensitivity_plots

    df = _toy_aggregates()
    csv_path = tmp_path / "strategy_aggregates.csv"
    df.to_csv(csv_path, index=False)
    paths = make_all_sensitivity_plots(tmp_path, n_sweep_points=5)
    for name in ("tornado", "stability", "sweep"):
        p = paths[name]
        assert p.exists(), f"Figur {name} ble ikke skrevet"
        assert p.stat().st_size > 1000, f"Figur {name} er suspekt liten"

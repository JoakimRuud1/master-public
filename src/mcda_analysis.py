"""Multi-Criteria Decision Analysis (MCDA) for the prompt-strategy comparison.

Aggregerer PDSQI-9-skårene per promptoppsett til en samlet, vektet kvalitetsskår
(Simple Additive Weighting / vektet sum), slik metode-kapittelet foreskriver.

Designvalg (forankret i kapittel 02 og 04 i masteroppgaven):
- Normalisering: min-max på den teoretiske skalaen 1-5 -> (x-1)/4, slik at
  alle kriterier blir sammenlignbare uten at variasjonen i datasettet selv
  påvirker normaliseringen.
- Vektet sum med vekter som reflekterer at klinisk korrekthet, dekning og
  nytte (Accurate, Thorough, Useful, Citation) er viktigere enn rent
  språklige kriterier (Comprehensible, Succinct). Vektene kan overstyres
  via CLI eller en JSON-fil.
- Synthesized er en betinget dimensjon (bare relevant naar abstraksjon = 1).
  Vi bruker promptens betingede gjennomsnitt (samme verdi som
  ``mean_synthesized_when_applicable`` i strategy_aggregates.csv), og om en
  prompt skulle mangle slike samtaler bortfaller dimensjonen og
  vektene renormaliseres.
- Stigmatiserende språk inngår ikke i PDSQI-9-vektene, men kan
  rapporteres som en valgfri penalty (introduced/propagated).
- Sensitivitetsanalyse: hver vekt perturberes +/-25 % mens de øvrige
  vektene holdes proporsjonalt konstante og normaliseres til sum 1.

Eksempel paa bruk:
    python -m src.mcda_analysis runs/20260426_142136
    python -m src.mcda_analysis runs/20260426_142136 --weights configs/mcda_weights.json
    python -m src.mcda_analysis runs/20260426_142136 --no-sensitivity
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

try:
    from src.judge_schema import (
        QUALITY_SCORE_KEYS,
        SYNTHESIS_SCORE_KEY,
    )
except Exception:  # pragma: no cover - skript-modus
    from judge_schema import (  # type: ignore
        QUALITY_SCORE_KEYS,
        SYNTHESIS_SCORE_KEY,
    )


# ---------------------------------------------------------------------------
# Konfigurasjon
# ---------------------------------------------------------------------------

# Standardvekter med forankring i teori- og metodekapittelet.
# - Accurate, Thorough, Useful og Citation faar hoeyere vekt enn rent
#   spraaklige kriterier (jf. seksjon 4.5 i oppgaven).
# - Synthesized faar lavere vekt fordi den bare gjelder en delmengde.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "accurate": 0.20,
    "thorough": 0.18,
    "useful": 0.18,
    "citation": 0.16,
    "organized": 0.10,
    "comprehensible": 0.08,
    "succinct": 0.06,
    "synthesized": 0.04,
}

# Disse kolonnene leses direkte fra strategy_aggregates.csv.
MEAN_COLUMN_BY_DIMENSION: Dict[str, str] = {
    **{k: f"mean_{k}" for k in QUALITY_SCORE_KEYS},
    SYNTHESIS_SCORE_KEY: "mean_synthesized_when_applicable",
}

# PDSQI-9 brukes paa skalaen 1..5 (teoretisk skala).
SCORE_MIN = 1.0
SCORE_MAX = 5.0


@dataclass
class MCDAConfig:
    """Innstillinger for MCDA-kjoeringen."""

    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    sensitivity_pct: float = 0.25  # +/-25 %
    apply_voice_penalty: bool = False
    voice_penalty: float = 0.05  # straff per "introduced" rate-poeng
    voice_penalty_column: str = "introduced_stigmatizing_language_rate"

    def normalised_weights(self) -> Dict[str, float]:
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("Vektene maa summere til en positiv verdi.")
        return {k: v / total for k, v in self.weights.items()}


# ---------------------------------------------------------------------------
# Innlesing / forberedelse
# ---------------------------------------------------------------------------


def load_strategy_aggregates(path: Path) -> pd.DataFrame:
    """Leser ``strategy_aggregates.csv`` fra en run-mappe."""
    csv_path = path if path.suffix == ".csv" else path / "strategy_aggregates.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Fant ikke strategy_aggregates.csv paa {csv_path}. "
            f"Forventet at <run-mappe>/strategy_aggregates.csv eksisterer."
        )
    df = pd.read_csv(csv_path)
    return df


def normalise_score(value: float, lo: float = SCORE_MIN, hi: float = SCORE_MAX) -> float:
    """Min-max-normaliserer en Likert-skaar paa 1..5 til [0, 1]."""
    if pd.isna(value):
        return float("nan")
    return (float(value) - lo) / (hi - lo)


def build_normalised_matrix(df: pd.DataFrame, dimensions: Sequence[str]) -> pd.DataFrame:
    """Bygger en (strategy x dimension)-matrise normalisert til [0, 1]."""
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        out: Dict[str, Any] = {"strategy_id": row["strategy_id"]}
        for dim in dimensions:
            col = MEAN_COLUMN_BY_DIMENSION[dim]
            if col not in df.columns:
                out[dim] = float("nan")
                continue
            out[dim] = normalise_score(row[col])
        rows.append(out)
    return pd.DataFrame(rows).set_index("strategy_id")


# ---------------------------------------------------------------------------
# Kjernen: vektet sum + sensitivitet
# ---------------------------------------------------------------------------


def weighted_score(
    normalised: pd.Series,
    weights: Mapping[str, float],
) -> Tuple[float, Dict[str, float]]:
    """Beregner samlet, vektet kvalitetsskaar for en strategi.

    Hvis en dimensjon mangler verdi for denne strategien (f.eks. ingen
    samtaler med abstraksjon=1), bortfaller dimensjonen og de oevrige
    vektene renormaliseres slik at de fortsatt summerer til 1.
    Returnerer (samlet skaar, brukt vektsett).
    """
    available = {dim: w for dim, w in weights.items() if not pd.isna(normalised.get(dim))}
    total = sum(available.values())
    if total <= 0:
        return float("nan"), {}
    used = {dim: w / total for dim, w in available.items()}
    score = sum(normalised[dim] * w for dim, w in used.items())
    return score, used


def compute_mcda_table(
    df: pd.DataFrame,
    config: MCDAConfig,
) -> pd.DataFrame:
    """Beregner samlet MCDA-skaar per promptoppsett.

    Returnerer en DataFrame sortert etter ``mcda_score`` synkende, med
    kolonner for hver normalisert dimensjon, vektet bidrag, samt rang.
    """
    weights = config.normalised_weights()
    dimensions = list(weights.keys())
    norm = build_normalised_matrix(df, dimensions)

    rows: List[Dict[str, Any]] = []
    for strat_id, norm_row in norm.iterrows():
        score, used = weighted_score(norm_row, weights)
        record: Dict[str, Any] = {
            "strategy_id": strat_id,
            "mcda_score": score,
            "weights_used_sum": sum(used.values()) if used else float("nan"),
            "n_dimensions_used": len(used),
        }
        # Lagre bidragene per dimensjon (vekt * normalisert skaar)
        for dim in dimensions:
            record[f"norm_{dim}"] = norm_row[dim]
            record[f"contrib_{dim}"] = (
                used[dim] * norm_row[dim] if dim in used else float("nan")
            )
        # Eventuell stigmatiserings-penalty
        if config.apply_voice_penalty and config.voice_penalty_column in df.columns:
            voice_rate = float(
                df.loc[df["strategy_id"] == strat_id, config.voice_penalty_column].iloc[0]
            )
            penalty = config.voice_penalty * voice_rate
            record["voice_rate"] = voice_rate
            record["voice_penalty"] = penalty
            record["mcda_score_with_penalty"] = score - penalty
        rows.append(record)

    out = pd.DataFrame(rows).sort_values("mcda_score", ascending=False).reset_index(drop=True)
    out.insert(1, "rank", out["mcda_score"].rank(ascending=False, method="min").astype(int))
    return out


# ---------------------------------------------------------------------------
# Sensitivitetsanalyse
# ---------------------------------------------------------------------------


def perturb_weights(
    base_weights: Mapping[str, float],
    target: str,
    factor: float,
) -> Dict[str, float]:
    """Skalerer vekten paa ``target`` med ``factor`` og renormaliserer resten
    proporsjonalt slik at summen fortsatt er 1.
    """
    base = dict(base_weights)
    if target not in base:
        raise KeyError(f"Ukjent vektkriterium: {target}")

    base_total = sum(base.values())
    if base_total <= 0:
        raise ValueError("Basevektene maa summere til en positiv verdi.")

    # Normaliser foerst til sum 1
    normalised = {k: v / base_total for k, v in base.items()}
    new_target = normalised[target] * factor
    others_sum = 1.0 - normalised[target]
    new_others_total = 1.0 - new_target
    perturbed: Dict[str, float] = {}
    for k, v in normalised.items():
        if k == target:
            perturbed[k] = new_target
        else:
            if others_sum > 0:
                perturbed[k] = v / others_sum * new_others_total
            else:
                perturbed[k] = 0.0
    return perturbed


def sensitivity_analysis(
    df: pd.DataFrame,
    config: MCDAConfig,
) -> pd.DataFrame:
    """Returnerer en oversikt over hvordan rangeringen endres naar hver
    enkelt vekt perturberes med +/-sensitivity_pct.
    """
    base_table = compute_mcda_table(df, config)
    base_rank = base_table.set_index("strategy_id")["rank"].to_dict()

    rows: List[Dict[str, Any]] = []
    for dim in config.weights:
        for direction, factor in (("+", 1 + config.sensitivity_pct), ("-", 1 - config.sensitivity_pct)):
            perturbed = perturb_weights(config.weights, dim, factor)
            new_cfg = MCDAConfig(
                weights=perturbed,
                sensitivity_pct=config.sensitivity_pct,
                apply_voice_penalty=config.apply_voice_penalty,
                voice_penalty=config.voice_penalty,
                voice_penalty_column=config.voice_penalty_column,
            )
            new_table = compute_mcda_table(df, new_cfg)
            new_rank = new_table.set_index("strategy_id")["rank"].to_dict()
            n_changes = sum(1 for s in base_rank if base_rank[s] != new_rank[s])
            top1_changed = (
                base_table.iloc[0]["strategy_id"] != new_table.iloc[0]["strategy_id"]
            )
            top3_set = set(base_table.head(3)["strategy_id"])
            new_top3 = set(new_table.head(3)["strategy_id"])
            top3_changed = top3_set != new_top3
            rows.append(
                {
                    "perturbed_weight": dim,
                    "direction": direction,
                    "factor": factor,
                    "n_rank_changes": n_changes,
                    "top1_changed": top1_changed,
                    "top3_changed": top3_changed,
                    "new_top1": new_table.iloc[0]["strategy_id"],
                    "new_top3": ",".join(new_table.head(3)["strategy_id"].tolist()),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sammenligning mot ren totalskaar (mean_primary_score)
# ---------------------------------------------------------------------------


def compare_with_primary_score(
    df: pd.DataFrame,
    mcda_table: pd.DataFrame,
) -> pd.DataFrame:
    """Sammenligner MCDA-rangeringen med rangeringen fra ``mean_primary_score``."""
    primary = (
        df[["strategy_id", "mean_primary_score"]]
        .copy()
        .sort_values("mean_primary_score", ascending=False)
        .reset_index(drop=True)
    )
    primary["primary_rank"] = primary["mean_primary_score"].rank(
        ascending=False, method="min"
    ).astype(int)
    out = mcda_table[["strategy_id", "rank", "mcda_score"]].merge(
        primary, on="strategy_id", how="left"
    )
    out = out.rename(columns={"rank": "mcda_rank"})
    out["rank_delta"] = out["primary_rank"] - out["mcda_rank"]
    return out.sort_values("mcda_rank")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_weights(weights_arg: Optional[str]) -> Dict[str, float]:
    """Leser vekter enten fra en JSON-fil eller fra en CLI-streng som
    ``accurate=0.2,thorough=0.18,...``.
    """
    if not weights_arg:
        return dict(DEFAULT_WEIGHTS)
    path = Path(weights_arg)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Vektfil maa vaere et JSON-objekt {dimensjon: vekt}.")
        return {k: float(v) for k, v in data.items()}
    # Tolk som ``key=val,key=val``
    out: Dict[str, float] = {}
    for chunk in weights_arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Forventet 'dim=verdi', fikk: {chunk!r}")
        k, v = chunk.split("=", 1)
        out[k.strip()] = float(v)
    return out


def write_weight_documentation(
    out_dir: Path,
    config: MCDAConfig,
    note: Optional[str] = None,
) -> None:
    """Skriver ut vektene som ble brukt, slik at MCDA-kjoeringen er
    etterprøvbar i tråd med metode-kapittelet.
    """
    payload = {
        "weights": config.weights,
        "weights_normalised": config.normalised_weights(),
        "sensitivity_pct": config.sensitivity_pct,
        "voice_penalty": {
            "active": config.apply_voice_penalty,
            "penalty_per_rate_unit": config.voice_penalty,
            "column": config.voice_penalty_column,
        },
        "note": note
        or (
            "Vektene reflekterer at klinisk korrekthet (Accurate), "
            "innholdsfullstendighet (Thorough), klinisk nytte (Useful) "
            "og kildeforankring (Citation) prioriteres hoeyere enn "
            "rent spraaklige kriterier, jf. seksjon 4.5 og kapittel 2."
        ),
    }
    (out_dir / "mcda_weights.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run_mcda(
    run_dir: Path,
    weights: Optional[Mapping[str, float]] = None,
    *,
    apply_voice_penalty: bool = False,
    sensitivity_pct: float = 0.25,
    do_sensitivity: bool = True,
    out_dir: Optional[Path] = None,
    weight_note: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Hoved-API: kjoerer MCDA-analysen for en run-mappe og lagrer csv-er.

    Returnerer et dict med tabellene 'mcda_table', 'comparison' og
    (valgfritt) 'sensitivity'.
    """
    df = load_strategy_aggregates(run_dir)
    config = MCDAConfig(
        weights=dict(weights) if weights else dict(DEFAULT_WEIGHTS),
        sensitivity_pct=sensitivity_pct,
        apply_voice_penalty=apply_voice_penalty,
    )

    out_dir = out_dir or run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mcda_table = compute_mcda_table(df, config)
    comparison = compare_with_primary_score(df, mcda_table)

    mcda_table.to_csv(out_dir / "mcda_results.csv", index=False)
    comparison.to_csv(out_dir / "mcda_vs_primary_score.csv", index=False)
    write_weight_documentation(out_dir, config, note=weight_note)

    results: Dict[str, pd.DataFrame] = {
        "mcda_table": mcda_table,
        "comparison": comparison,
    }

    if do_sensitivity:
        sens = sensitivity_analysis(df, config)
        sens.to_csv(out_dir / "mcda_sensitivity.csv", index=False)
        results["sensitivity"] = sens

    return results


def _format_for_console(df: pd.DataFrame, columns: Sequence[str]) -> str:
    return df[list(columns)].to_string(index=False, float_format=lambda v: f"{v:0.4f}")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Vektet MCDA over PDSQI-9-skaarene.")
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Sti til run-mappe (eller direkte til strategy_aggregates.csv).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help=(
            "Sti til JSON-fil eller komma-separert streng paa formen "
            "'accurate=0.2,thorough=0.18,...'. Hvis ikke oppgitt brukes "
            "DEFAULT_WEIGHTS."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Hvor MCDA-tabellene skal lagres. Default: samme run-mappe.",
    )
    parser.add_argument(
        "--apply-voice-penalty",
        action="store_true",
        help="Trekker fra en penalty for andelen sammendrag som introduserte stigmatiserende språk.",
    )
    parser.add_argument(
        "--sensitivity-pct",
        type=float,
        default=0.25,
        help="Stoerrelse paa vekt-perturbasjonen (default 0.25 = +/-25%%).",
    )
    parser.add_argument(
        "--no-sensitivity",
        action="store_true",
        help="Hopp over sensitivitetsanalysen.",
    )
    parser.add_argument(
        "--note",
        type=str,
        default=None,
        help="Fri tekst-begrunnelse som lagres sammen med vektene.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    run_dir = args.run_dir
    weights = parse_weights(args.weights) if args.weights else None

    results = run_mcda(
        run_dir,
        weights=weights,
        apply_voice_penalty=args.apply_voice_penalty,
        sensitivity_pct=args.sensitivity_pct,
        do_sensitivity=not args.no_sensitivity,
        out_dir=args.out_dir,
        weight_note=args.note,
    )

    print("\n=== MCDA-rangering (vektet sum, normalisert paa 1-5) ===\n")
    print(
        _format_for_console(
            results["mcda_table"],
            ["rank", "strategy_id", "mcda_score", "n_dimensions_used"],
        )
    )

    print("\n=== MCDA vs. mean_primary_score ===\n")
    print(
        _format_for_console(
            results["comparison"],
            [
                "mcda_rank",
                "primary_rank",
                "rank_delta",
                "strategy_id",
                "mcda_score",
                "mean_primary_score",
            ],
        )
    )

    if "sensitivity" in results:
        sens = results["sensitivity"]
        n_top1 = int(sens["top1_changed"].sum())
        n_top3 = int(sens["top3_changed"].sum())
        print("\n=== Sensitivitet (+/-{:.0f} %) ===".format(args.sensitivity_pct * 100))
        print(
            f"Topp-1 endret seg i {n_top1} av {len(sens)} perturbasjoner; "
            f"topp-3 endret seg i {n_top3}."
        )
        if n_top1 + n_top3 > 0:
            print("\nDetaljer for perturbasjoner som endret topp-3 eller topp-1:")
            changed = sens[sens["top1_changed"] | sens["top3_changed"]]
            print(
                _format_for_console(
                    changed,
                    [
                        "perturbed_weight",
                        "direction",
                        "n_rank_changes",
                        "top1_changed",
                        "top3_changed",
                        "new_top1",
                    ],
                )
            )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

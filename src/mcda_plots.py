"""Visualiseringer for MCDA-sensitivitetsanalysen.

Tre figurer som er standard i MCDA-rapportering:

1. ``plot_tornado``: Hvor mye topp-strategiens skaar endrer seg naar hver
   vekt perturberes +/-25 %. Dette viser hvilke vekter rangeringen er
   foelsom for (lange staver = innflytelsesrike vekter).
2. ``plot_rank_stability_matrix``: En heatmap over rangeringsendringer per
   strategi paa tvers av alle perturbasjoner. Viser hvor stabilt hvert
   promptoppsett er rangert.
3. ``plot_weight_sweep``: For hver vekt, varier den fra 0 til 1 mens de
   oevrige holdes proporsjonalt konstante, og plott samlet skaar per
   strategi som funksjon av vekten. Viser hvor mye en vekt maa endres
   foer rangeringen "knekker".

Eksempel:
    python -m src.mcda_plots runs/20260426_142136
    python -m src.mcda_plots runs/20260426_142136 --weights configs/mcda_weights.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from src.mcda_analysis import (
        DEFAULT_WEIGHTS,
        MCDAConfig,
        compute_mcda_table,
        load_strategy_aggregates,
        parse_weights,
        perturb_weights,
    )
except Exception:  # pragma: no cover - skript-modus
    from mcda_analysis import (  # type: ignore
        DEFAULT_WEIGHTS,
        MCDAConfig,
        compute_mcda_table,
        load_strategy_aggregates,
        parse_weights,
        perturb_weights,
    )


# Konsekvent fargepalett — kategoriske farger som er greie i print
_CMAP = "tab20"


def _short_label(strategy_id: str, max_len: int = 28) -> str:
    label = strategy_id.replace("_", " ")
    return label if len(label) <= max_len else label[: max_len - 1] + "…"


# ---------------------------------------------------------------------------
# Hjelpere for sensitivitet
# ---------------------------------------------------------------------------


def _scores_for_weights(
    df: pd.DataFrame,
    weights: Mapping[str, float],
    base_config: MCDAConfig,
) -> pd.Series:
    """Beregner MCDA-skaar per strategi for et gitt vektsett."""
    cfg = MCDAConfig(
        weights=dict(weights),
        sensitivity_pct=base_config.sensitivity_pct,
        apply_voice_penalty=base_config.apply_voice_penalty,
        voice_penalty=base_config.voice_penalty,
        voice_penalty_column=base_config.voice_penalty_column,
    )
    table = compute_mcda_table(df, cfg)
    return table.set_index("strategy_id")["mcda_score"]


def _ranks_for_weights(
    df: pd.DataFrame,
    weights: Mapping[str, float],
    base_config: MCDAConfig,
) -> pd.Series:
    cfg = MCDAConfig(
        weights=dict(weights),
        sensitivity_pct=base_config.sensitivity_pct,
        apply_voice_penalty=base_config.apply_voice_penalty,
        voice_penalty=base_config.voice_penalty,
        voice_penalty_column=base_config.voice_penalty_column,
    )
    table = compute_mcda_table(df, cfg)
    return table.set_index("strategy_id")["rank"]


# ---------------------------------------------------------------------------
# 1. Tornado-plot
# ---------------------------------------------------------------------------


def plot_tornado(
    df: pd.DataFrame,
    config: MCDAConfig,
    out_path: Path,
    *,
    target_strategy: Optional[str] = None,
) -> Path:
    """Tegner et tornado-plot for hvor mye topp-strategiens skaar endrer
    seg naar hver vekt perturberes +/-sensitivity_pct.
    """
    base_scores = _scores_for_weights(df, config.weights, config)
    base_table = compute_mcda_table(df, config)
    target = target_strategy or base_table.iloc[0]["strategy_id"]
    base_score = base_scores[target]

    deltas: List[Dict[str, float]] = []
    for dim in config.weights:
        plus = perturb_weights(config.weights, dim, 1 + config.sensitivity_pct)
        minus = perturb_weights(config.weights, dim, 1 - config.sensitivity_pct)
        s_plus = _scores_for_weights(df, plus, config)[target] - base_score
        s_minus = _scores_for_weights(df, minus, config)[target] - base_score
        deltas.append({"dimension": dim, "plus": s_plus, "minus": s_minus})

    deltas.sort(key=lambda d: max(abs(d["plus"]), abs(d["minus"])), reverse=False)
    dims = [d["dimension"] for d in deltas]
    pluses = [d["plus"] for d in deltas]
    minuses = [d["minus"] for d in deltas]

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(dims) + 1.8))
    y = np.arange(len(dims))
    ax.barh(y, pluses, color="#3a78b5", label=f"+{config.sensitivity_pct*100:.0f}% vekt")
    ax.barh(y, minuses, color="#c44e52", label=f"-{config.sensitivity_pct*100:.0f}% vekt")
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(dims)
    ax.set_xlabel(f"Endring i MCDA-skaar for «{_short_label(target)}»")
    ax.set_title(
        f"Tornado: foelsomhet i topp-strategiens skaar\n"
        f"(referansestrategi: {_short_label(target)})"
    )
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 2. Rangeringsstabilitetsmatrise
# ---------------------------------------------------------------------------


def plot_rank_stability_matrix(
    df: pd.DataFrame,
    config: MCDAConfig,
    out_path: Path,
) -> Path:
    """Heatmap som viser rang per strategi (rader) under base + alle
    +/-perturbasjoner (kolonner). Cellefarge = rang; tall = rang.
    """
    base_ranks = _ranks_for_weights(df, config.weights, config)
    strategies = base_ranks.sort_values().index.tolist()  # rang 1 -> 8

    columns: List[str] = ["base"]
    rank_matrix: Dict[str, pd.Series] = {"base": base_ranks}

    for dim in config.weights:
        for sign, factor in (("+", 1 + config.sensitivity_pct), ("-", 1 - config.sensitivity_pct)):
            label = f"{dim}{sign}"
            perturbed = perturb_weights(config.weights, dim, factor)
            rank_matrix[label] = _ranks_for_weights(df, perturbed, config)
            columns.append(label)

    matrix = pd.DataFrame({c: rank_matrix[c] for c in columns}).reindex(strategies)

    fig, ax = plt.subplots(
        figsize=(0.55 * len(columns) + 2.5, 0.55 * len(strategies) + 1.5)
    )
    n_strats = len(strategies)
    im = ax.imshow(matrix.values, cmap="RdYlGn_r", vmin=1, vmax=n_strats, aspect="auto")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels([_short_label(s) for s in strategies], fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = int(matrix.iat[i, j])
            ax.text(j, i, str(v), ha="center", va="center", fontsize=7, color="black")
    ax.set_title(
        f"Rangeringsstabilitet under +/-{config.sensitivity_pct*100:.0f}% vektperturbasjoner"
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Rang (1 = best)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 3. Kontinuerlig vekt-sweep per dimensjon
# ---------------------------------------------------------------------------


def _sweep_one_weight(
    df: pd.DataFrame,
    config: MCDAConfig,
    target: str,
    sweep_values: Sequence[float],
) -> pd.DataFrame:
    """Returnerer en DataFrame: indeks = sweep-verdi, kolonner = strategier,
    verdier = MCDA-skaar.
    """
    base = config.normalised_weights()
    others_total = 1.0 - base[target]

    rows: Dict[float, Dict[str, float]] = {}
    for w in sweep_values:
        new_weights: Dict[str, float] = {}
        new_weights[target] = float(w)
        new_others_total = 1.0 - float(w)
        for k, v in base.items():
            if k == target:
                continue
            if others_total > 0:
                new_weights[k] = v / others_total * new_others_total
            else:
                new_weights[k] = 0.0
        scores = _scores_for_weights(df, new_weights, config)
        rows[float(w)] = scores.to_dict()
    return pd.DataFrame(rows).T


def plot_weight_sweep(
    df: pd.DataFrame,
    config: MCDAConfig,
    out_path: Path,
    *,
    n_points: int = 21,
    dimensions: Optional[Sequence[str]] = None,
) -> Path:
    """Tegner samlet skaar per strategi som funksjon av hver vekt
    (de oevrige skaleres proporsjonalt). Lager ett subplot per dimensjon.
    """
    dims = list(dimensions) if dimensions else list(config.weights.keys())
    sweep_values = np.linspace(0.0, 1.0, n_points)

    n_cols = 3 if len(dims) <= 9 else 4
    n_rows = int(np.ceil(len(dims) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.6 * n_cols, 3.2 * n_rows), sharey=True
    )
    axes = np.atleast_2d(axes)

    base_norm_weights = config.normalised_weights()
    base_table = compute_mcda_table(df, config)
    strategies = base_table["strategy_id"].tolist()
    cmap = plt.get_cmap(_CMAP, max(len(strategies), 8))

    for idx, dim in enumerate(dims):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        sweep_df = _sweep_one_weight(df, config, dim, sweep_values)
        for i, strat in enumerate(strategies):
            ax.plot(
                sweep_df.index,
                sweep_df[strat],
                color=cmap(i),
                linewidth=1.4,
                label=_short_label(strat) if idx == 0 else None,
            )
        # Marker basevekten med en vertikal stiplet linje
        ax.axvline(
            base_norm_weights[dim],
            color="black",
            linestyle=":",
            linewidth=0.9,
            label="basevekt" if idx == 0 else None,
        )
        ax.set_title(dim, fontsize=10)
        ax.set_xlabel(f"Vekt for {dim}", fontsize=9)
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=8)
        if c == 0:
            ax.set_ylabel("MCDA-skaar", fontsize=9)

    # Skjul ubrukte aksene
    for idx in range(len(dims), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(4, len(labels)),
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Vekt-sweep: samlet skaar som funksjon av hver enkelt vekt\n"
        "(de oevrige vektene skaleres proporsjonalt slik at sum = 1)",
        fontsize=11,
    )
    plt.tight_layout(rect=(0, 0.04, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Hoved-API
# ---------------------------------------------------------------------------


def make_all_sensitivity_plots(
    run_dir: Path,
    *,
    weights: Optional[Mapping[str, float]] = None,
    sensitivity_pct: float = 0.25,
    apply_voice_penalty: bool = False,
    out_dir: Optional[Path] = None,
    n_sweep_points: int = 21,
) -> Dict[str, Path]:
    """Genererer alle tre sensitivitetsfigurene og returnerer stiene."""
    df = load_strategy_aggregates(run_dir)
    config = MCDAConfig(
        weights=dict(weights) if weights else dict(DEFAULT_WEIGHTS),
        sensitivity_pct=sensitivity_pct,
        apply_voice_penalty=apply_voice_penalty,
    )
    out_dir = out_dir or (run_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "tornado": plot_tornado(df, config, out_dir / "mcda_tornado.png"),
        "stability": plot_rank_stability_matrix(
            df, config, out_dir / "mcda_rank_stability.png"
        ),
        "sweep": plot_weight_sweep(
            df, config, out_dir / "mcda_weight_sweep.png", n_points=n_sweep_points
        ),
    }
    return paths


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Visualiseringer for MCDA-sensitivitetsanalysen."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--sensitivity-pct", type=float, default=0.25)
    parser.add_argument("--apply-voice-penalty", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--sweep-points", type=int, default=21)
    args = parser.parse_args(list(argv) if argv is not None else None)

    weights = parse_weights(args.weights) if args.weights else None
    paths = make_all_sensitivity_plots(
        args.run_dir,
        weights=weights,
        sensitivity_pct=args.sensitivity_pct,
        apply_voice_penalty=args.apply_voice_penalty,
        out_dir=args.out_dir,
        n_sweep_points=args.sweep_points,
    )
    for name, p in paths.items():
        print(f"{name}: {p}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

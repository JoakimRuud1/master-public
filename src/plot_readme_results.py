import csv
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "docs" / "readme_dimension_scores.csv"
OUTPUT_PATH = ROOT / "docs" / "prompt_strategy_dimension_deltas.svg"
MAX_ABS_DELTA = 0.45


def blend(start: tuple[int, int, int], end: tuple[int, int, int], amount: float) -> str:
    amount = max(0.0, min(1.0, amount))
    channels = [round(a + (b - a) * amount) for a, b in zip(start, end)]
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def cell_color(delta: float) -> str:
    neutral = (247, 247, 247)
    target = (47, 125, 101) if delta >= 0 else (190, 80, 80)
    return blend(neutral, target, abs(delta) / MAX_ABS_DELTA)


def text(x: float, y: float, value: str, **attributes: object) -> str:
    attrs = " ".join(f'{key.replace("_", "-")}="{escape(str(item))}"' for key, item in attributes.items())
    return f'<text x="{x}" y="{y}" {attrs}>{escape(value)}</text>'


def main() -> None:
    """Plot rounded thesis results as score differences from the baseline."""
    with DATA_PATH.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    dimensions = [column for column in rows[0] if column != "strategy"]
    baseline = {dimension: float(rows[0][dimension]) for dimension in dimensions}

    width, height = 1220, 610
    left, top = 330, 120
    cell_width, cell_height = 118, 52
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<g font-family="Arial, Helvetica, sans-serif" fill="#1f2933">',
        text(24, 36, "Prompt strategy performance by evaluation dimension", font_size=23, font_weight=700),
        text(24, 63, "Mean score difference from the minimal baseline (1-5 scale)", font_size=15, fill="#52606d"),
    ]

    for column_index, dimension in enumerate(dimensions):
        x = left + column_index * cell_width + cell_width / 2
        words = dimension.split()
        if len(words) > 1:
            svg.append(text(x, 88, words[0], text_anchor="middle", font_size=13, font_weight=600))
            svg.append(text(x, 105, " ".join(words[1:]), text_anchor="middle", font_size=13, font_weight=600))
        else:
            svg.append(text(x, 98, dimension, text_anchor="middle", font_size=13, font_weight=600))

    for row_index, row in enumerate(rows):
        y = top + row_index * cell_height
        svg.append(text(left - 14, y + 33, row["strategy"], text_anchor="end", font_size=14))
        for column_index, dimension in enumerate(dimensions):
            delta = float(row[dimension]) - baseline[dimension]
            x = left + column_index * cell_width
            svg.append(
                f'<rect x="{x}" y="{y}" width="{cell_width - 2}" height="{cell_height - 2}" '
                f'fill="{cell_color(delta)}" rx="2"/>'
            )
            label = "0.00" if abs(delta) < 0.005 else f"{delta:+.2f}"
            svg.append(text(x + (cell_width - 2) / 2, y + 32, label, text_anchor="middle", font_size=14, font_weight=600))

    legend_y = top + len(rows) * cell_height + 28
    legend_values = [(-0.40, "-0.40"), (0.00, "0.00"), (0.40, "+0.40")]
    svg.append(text(left, legend_y + 17, "Color scale:", font_size=13, fill="#52606d"))
    for index, (value, label) in enumerate(legend_values):
        x = left + 92 + index * 92
        svg.append(f'<rect x="{x}" y="{legend_y}" width="26" height="20" fill="{cell_color(value)}" rx="2"/>')
        svg.append(text(x + 34, legend_y + 16, label, font_size=12, fill="#52606d"))

    svg.extend(["</g>", "</svg>"])
    OUTPUT_PATH.write_text("\n".join(svg) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
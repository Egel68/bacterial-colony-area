import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _load_run_data(run_dir: Path) -> tuple[dict, list, list]:
    with open(run_dir / "summary.json") as f:
        summary = json.load(f)
    val_log = summary.get("val_log", [])
    train_log = summary.get("train_log", [])
    return summary, train_log, val_log


def _make_metrics_plot(train_hist: list[dict], val_hist: list[dict], metric: str) -> go.Figure:
    fig = go.Figure()
    epochs = list(range(1, len(train_hist) + 1))
    fig.add_trace(go.Scatter(x=epochs, y=[m[metric] for m in train_hist], mode="lines", name=f"train_{metric}"))
    fig.add_trace(go.Scatter(x=epochs, y=[m[metric] for m in val_hist], mode="lines", name=f"val_{metric}"))
    fig.update_layout(title=f"{metric.upper()} over epochs", xaxis_title="epoch", yaxis_title=metric)
    return fig


def generate_report(summary: dict, train_hist: list[dict], val_hist: list[dict], output_path: Path):
    figures = []
    for metric in ["loss", "iou", "dice", "precision", "recall"]:
        fig = _make_metrics_plot(train_hist, val_hist, metric)
        figures.append(fig.to_html(full_html=False, include_plotlyjs=False))

    metrics_html = ""
    for k, v in summary.items():
        if k not in ("train_log", "val_log"):
            metrics_html += f"<tr><td>{k}</td><td>{v}</td></tr>\n"

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Training Report - {summary.get("model", "unknown")}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
body {{ font-family: sans-serif; margin: 2rem; background: #f5f5f5; }}
h1, h2 {{ color: #333; }}
table {{ border-collapse: collapse; margin: 1rem 0; }}
td, th {{ border: 1px solid #ccc; padding: 0.5rem 1rem; text-align: left; }}
.plot {{ background: #fff; padding: 1rem; border-radius: 8px; margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
</style>
</head>
<body>
<h1>Training Report</h1>
<table>
{metrics_html}
</table>
<h2>Metrics</h2>
{"".join(f'<div class="plot">{f}</div>' for f in figures)}
</body>
</html>"""
    output_path.write_text(html)

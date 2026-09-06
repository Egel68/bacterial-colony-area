import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from .runner import (
    AllResults,
    compute_all_winners,
    compute_detailed_metrics,
    compute_outlier_table,
    compute_summary,
    compute_wilcoxon_table,
)
from .statistics import compute_descriptive, detect_outliers

log = logging.getLogger(__name__)

METRIC_KEYS = ["iou", "dice", "f1", "precision", "recall", "accuracy"]
METRIC_LABELS = ["IoU", "Dice", "F1", "Precision", "Recall", "Accuracy"]


def _stat_header_cells() -> str:
    cols = ""
    for label in METRIC_LABELS:
        cols += f"""<th>{label}<br><small>Mean±Std</small></th>
    <th>{label}<br><small>Median</small></th>
    <th>{label}<br><small>Q1–Q3</small></th>
    <th>{label}<br><small>P5–P95</small></th>
    <th>{label}<br><small>Min–Max</small></th>"""
    return cols


def _stat_data_cells(entry: Dict) -> str:
    cells = ""
    for key in METRIC_KEYS:
        mean = entry.get(f"mean_{key}", 0)
        std = entry.get(f"std_{key}", 0)
        median = entry.get(f"median_{key}", 0)
        q1 = entry.get(f"q1_{key}", 0)
        q3 = entry.get(f"q3_{key}", 0)
        p5 = entry.get(f"p5_{key}", 0)
        p95 = entry.get(f"p95_{key}", 0)
        mn = entry.get(f"min_{key}", 0)
        mx = entry.get(f"max_{key}", 0)
        cells += f"""<td class="num">{mean:.4f} ± {std:.4f}</td>
    <td class="num">{median:.4f}</td>
    <td class="num">{q1:.4f} – {q3:.4f}</td>
    <td class="num">{p5:.4f} – {p95:.4f}</td>
    <td class="num">{mn:.4f} – {mx:.4f}</td>"""
    return cells


def _build_summary_rows(summary: List[Dict]) -> str:
    rows = ""
    for entry in summary:
        rows += f"""<tr>
  <td><b>{entry["name"]}</b></td>
  <td>{entry["description"]}</td>
  <td>{entry["num_images"]}</td>
  {_stat_data_cells(entry)}
</tr>\n"""
    return rows


def _build_per_algo_sections(all_results: AllResults) -> str:
    sections = ""
    for algo_name, algo_results in all_results.items():
        rows = ""
        for sample_key, variants in sorted(algo_results.items()):
            for variant_key in ("source", "cropped"):
                if variant_key not in variants:
                    continue
                m = variants[variant_key]
                vals = "".join(
                    f'<td class="num">{m.get(k, 0):.4f}</td>' for k in METRIC_KEYS
                )
                rows += f"""<tr>
  <td>{sample_key}</td>
  <td>{variant_key}</td>
  {vals}
</tr>\n"""

        sections += f"""<h2>{algo_name}</h2>
<table class="per-algo">
  <thead>
    <tr>
      <th>Image</th>
      <th>Type</th>
      {"".join(f"<th>{label}</th>" for label in METRIC_LABELS)}
    </tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>
"""
    return sections


def _build_chart_data(summary: List[Dict]) -> str:
    data = []
    for entry in summary:
        data.append(
            {
                "name": entry["name"],
                "iou": entry.get("mean_iou", 0),
                "dice": entry.get("mean_dice", 0),
                "f1": entry.get("mean_f1", 0),
                "precision": entry.get("mean_precision", 0),
                "recall": entry.get("mean_recall", 0),
            }
        )
    return json.dumps(data, ensure_ascii=False)


def _build_boxplot_data(summary: List[Dict]) -> str:
    """Формирует JSON с данными для box-plot по каждой метрике.

    Структура: { metric_key: { algo_name: { min, q1, median, q3, max, outliers[] } } }
    """
    data: Dict = {}
    for key in METRIC_KEYS:
        metric_data: Dict = {}
        for entry in summary:
            algo_name = entry["name"]
            vals = []
            # We don't have per-sample values in summary, so outlier detection
            # is approximate; the scatter chart has exact outliers.
            metric_data[algo_name] = {
                "min": entry.get(f"min_{key}", 0),
                "q1": entry.get(f"q1_{key}", 0),
                "median": entry.get(f"median_{key}", 0),
                "q3": entry.get(f"q3_{key}", 0),
                "max": entry.get(f"max_{key}", 0),
                "outliers": [],
            }
        data[key] = metric_data
    return json.dumps(data, ensure_ascii=False)


def _build_scatter_data(all_results: AllResults) -> str:
    """Формирует JSON scatter-данных per-sample для каждой метрики и алгоритма.

    Структура: { metric_key: { algo_name: [{ sample, value, is_outlier }] } }
    """
    data: Dict = {}
    for key in METRIC_KEYS:
        metric_data: Dict = {}
        for algo_name, algo_results in all_results.items():
            points = []
            vals = []
            meta = []
            for sample_key, variants in algo_results.items():
                for variant_key, m in variants.items():
                    val = m.get(key, 0.0)
                    vals.append(val)
                    meta.append((sample_key, variant_key))
            if vals:
                desc = compute_descriptive(vals)
                mean_v = desc["mean"]
                std_v = desc["std"]
                outlier_vals = {
                    o["value"]
                    for o in detect_outliers(vals)
                }
                for (sk, vk), val in zip(meta, vals):
                    points.append(
                        {
                            "sample": f"{sk}/{vk}",
                            "value": round(val, 4),
                            "is_outlier": val in outlier_vals,
                        }
                    )
            metric_data[algo_name] = points
        data[key] = metric_data
    return json.dumps(data, ensure_ascii=False)


def _build_significance_section(wilcoxon_table: Dict) -> str:
    sections = ""
    for metric_name, comparisons in wilcoxon_table.items():
        rows = ""
        for pair_key, result in sorted(comparisons.items()):
            p_str = f"{result['p_value']:.6f}" if result["p_value"] is not None else "—"
            interp_map = {
                "significant": "✅ значимо",
                "not_significant": "❌ не значимо",
                "identical": "＝ идентичны",
            }
            interp = interp_map.get(result["interpretation"], "—")
            rows += f"""<tr>
  <td>{pair_key.replace(' vs ', '</b> vs <b>')}</td>
  <td class="num">{p_str}</td>
  <td>{interp}</td>
</tr>\n"""
        sections += f"""<h2>Статистическая значимость: {metric_label(metric_name)}</h2>
<table class="significance">
  <thead>
    <tr>
      <th>Сравнение</th>
      <th>p-value (Wilcoxon)</th>
      <th>Интерпретация (α=0.05)</th>
    </tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>
"""
    return sections


def _build_winner_table(all_winners: Dict) -> str:
    sections = ""
    for metric_name, comparisons in all_winners.items():
        rows = ""
        for pair_key, fractions in sorted(comparisons.items()):
            a, b = pair_key.split(" vs ")
            rows += f"""<tr>
  <td><b>{a}</b></td>
  <td><b>{b}</b></td>
  <td class="num">{fractions['a_wins']:.1%}</td>
  <td class="num">{fractions['b_wins']:.1%}</td>
  <td class="num">{fractions['ties']:.1%}</td>
</tr>\n"""
        sections += f"""<h2>Доля побед по снимкам: {metric_label(metric_name)}</h2>
<table class="winner-table">
  <thead>
    <tr>
      <th>Алгоритм A</th>
      <th>Алгоритм B</th>
      <th>A победил</th>
      <th>B победил</th>
      <th>Ничья</th>
    </tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>
"""
    return sections


def _build_outlier_table(outliers: List[Dict]) -> str:
    if not outliers:
        return ""
    rows = ""
    for o in outliers:
        rows += f"""<tr>
  <td>{o['algorithm']}</td>
  <td>{metric_label(o['metric'])}</td>
  <td>{o['sample']}</td>
  <td>{o['variant']}</td>
  <td class="num">{o['value']:.4f}</td>
  <td class="num">{o['z_score']:.2f}</td>
</tr>\n"""
    return f"""<h2>Выбросы (|z| > 3σ)</h2>
<details>
  <summary>Показать выбросы ({len(outliers)})</summary>
  <table class="outlier-table">
    <thead>
      <tr>
        <th>Algorithm</th>
        <th>Metric</th>
        <th>Image</th>
        <th>Type</th>
        <th>Value</th>
        <th>Z-score</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</details>
"""


def _build_comparison_sections(
    comparison: Dict[str, Dict[str, Dict[str, str]]],
) -> str:
    sections = ""
    for metric_name, samples in sorted(comparison.items()):
        rows = ""
        for sample_key, variants in sorted(samples.items()):
            for variant, winner in sorted(variants.items()):
                winner_label = {
                    "a": "A",
                    "b": "B",
                    "tie": "Ничья",
                }.get(winner, "—")
                rows += f"""<tr>
  <td>{sample_key}</td>
  <td>{variant}</td>
  <td class="num"><b>{winner_label}</b></td>
</tr>\n"""
        sections += f"""<h2>Сравнение: {metric_label(metric_name)}</h2>
<table class="comparison">
  <thead>
    <tr>
      <th>Image</th>
      <th>Type</th>
      <th>Winner</th>
    </tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>
"""
    return sections


def metric_label(metric_name: str) -> str:
    labels = {
        "iou": "IoU",
        "dice": "Dice",
        "f1": "F1",
        "precision": "Precision",
        "recall": "Recall",
        "accuracy": "Accuracy",
    }
    return labels.get(metric_name, metric_name)


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Testing Report — Colony Detection</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #11111b; color: #cdd6f4; padding: 2rem; }}
h1 {{ color: #89b4fa; margin-bottom: 1.5rem; }}
h2 {{ color: #89b4fa; margin: 2rem 0 1rem; font-size: 1.2rem; }}
h3 {{ color: #a6adc8; margin: 1rem 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }}
th, td {{ padding: 0.5rem 0.75rem; text-align: left; border-bottom: 1px solid #313244; }}
th {{ background: #181825; color: #a6adc8; font-weight: 600; white-space: nowrap; }}
th small {{ color: #585b70; font-weight: 400; }}
tr:hover {{ background: #1e1e2e; }}
.num {{ text-align: right; font-family: "JetBrains Mono", "Fira Code", monospace; }}
.summary {{ background: #181825; border-radius: 8px; padding: 0; }}
.per-algo {{ margin-bottom: 2rem; }}
.comparison {{ margin-bottom: 2rem; }}
.comparison td b {{ color: #a6e3a1; }}
.significance td b {{ color: #f9e2af; }}
.winner-table td.num {{ color: #a6e3a1; }}
.outlier-table td.num {{ color: #f38ba8; }}
details {{ margin: 1rem 0; }}
details summary {{ cursor: pointer; color: #89b4fa; font-weight: 600; padding: 0.5rem; background: #181825; border-radius: 4px; }}
.chart-container {{ background: #181825; border-radius: 8px; padding: 1.5rem; margin: 1.5rem 0; }}
.chart-row {{ display: flex; gap: 1.5rem; flex-wrap: wrap; }}
.chart-box {{ flex: 1; min-width: 300px; }}
.boxplot-grid {{ display: flex; gap: 1.5rem; flex-wrap: wrap; }}
.boxplot-chart {{ flex: 1; min-width: 400px; background: #181825; border-radius: 8px; padding: 1rem; }}
.scatter-grid {{ display: flex; gap: 1.5rem; flex-wrap: wrap; }}
.scatter-chart {{ flex: 1; min-width: 350px; background: #181825; border-radius: 8px; padding: 1rem; }}
.footer {{ margin-top: 3rem; color: #585b70; font-size: 0.8rem; text-align: center; }}
</style>
</head>
<body>
<h1>Testing Report — Colony Detection</h1>

<div class="chart-container">
  <h3>Aggregate Metrics (Mean)</h3>
  <div class="chart-row">
    <div class="chart-box"><canvas id="chart_iou"></canvas></div>
    <div class="chart-box"><canvas id="chart_dice"></canvas></div>
  </div>
  <div class="chart-row">
    <div class="chart-box"><canvas id="chart_f1"></canvas></div>
    <div class="chart-box"><canvas id="chart_pr"></canvas></div>
  </div>
</div>

{boxplot_sections}

{scatter_sections}

<h2>Summary</h2>
<table class="summary">
  <thead>
    <tr>
      <th>Algorithm</th>
      <th>Description</th>
      <th>Images</th>
      {stat_headers}
    </tr>
  </thead>
  <tbody>
    {summary_rows}
  </tbody>
</table>

{significance_sections}

{winner_sections}

{outlier_section}

<hr style="border-color: #313244; margin: 2rem 0;">

<h2>Per-Algorithm Details</h2>
{per_algo_sections}

{comparison_sections}

<div class="footer">
Generated by bacterial-colony-area testing framework — {timestamp}
</div>

<script>
// ===== Chart.js plugins =====
const boxplotPlugin = {{
  id: 'boxplot',
  afterDraw(chart) {{
    const ctx = chart.ctx;
    const xAxis = chart.scales.x;
    const yAxis = chart.scales.y;
    if (!xAxis || !yAxis) return;
    const boxData = chart.config.data.boxData;
    if (!boxData) return;
    const labels = chart.data.labels;
    if (!labels || labels.length === 0) return;
    const barWidth = 0.6 / labels.length;
    const colors = ['#89b4fa', '#a6e3a1', '#f9e2af', '#fab387', '#cba6f7', '#f38ba8'];
    const tickWidth = xAxis.width / labels.length;

    labels.forEach((name, i) => {{
      const s = boxData[name];
      if (!s) return;
      const xCenter = xAxis.getPixelForValue(i);
      const x = xCenter;
      const yQ1 = yAxis.getPixelForValue(s.q1);
      const yQ3 = yAxis.getPixelForValue(s.q3);
      const yMed = yAxis.getPixelForValue(s.median);
      const yMin = yAxis.getPixelForValue(s.min);
      const yMax = yAxis.getPixelForValue(s.max);

      // Skip invalid positions
      if (!yQ1 || !yQ3) return;

      // Whisker line
      ctx.strokeStyle = '#cdd6f4';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x, yMin); ctx.lineTo(x, yMax);
      ctx.stroke();

      // Whisker caps
      ctx.strokeStyle = '#cdd6f4';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x - 6, yMin); ctx.lineTo(x + 6, yMin);
      ctx.moveTo(x - 6, yMax); ctx.lineTo(x + 6, yMax);
      ctx.stroke();

      // Box (Q1-Q3)
      const color = colors[i % colors.length];
      ctx.fillStyle = color + '80';
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      const boxHalf = Math.min(15, tickWidth * 0.2);
      ctx.fillRect(x - boxHalf, yQ1, boxHalf * 2, yQ3 - yQ1);
      ctx.strokeRect(x - boxHalf, yQ1, boxHalf * 2, yQ3 - yQ1);

      // Median line
      ctx.strokeStyle = '#11111b';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(x - boxHalf, yMed); ctx.lineTo(x + boxHalf, yMed);
      ctx.stroke();

      // Outliers
      if (s.outliers && s.outliers.length) {{
        ctx.fillStyle = '#f38ba8';
        s.outliers.forEach(o => {{
          const y = yAxis.getPixelForValue(o);
          if (!y) return;
          ctx.beginPath();
          ctx.arc(x, y, 3, 0, Math.PI * 2);
          ctx.fill();
        }});
      }}
    }});
  }}
}};

const chartData = {chart_data};
const boxplotData = {boxplot_data};
const scatterData = {scatter_data};
const COLORS = ['#89b4fa', '#a6e3a1', '#f9e2af', '#fab387', '#cba6f7', '#f38ba8'];

// ===== Aggregate bar charts =====
function makeBar(ctx, labels, values, label, color) {{
  new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels: labels,
      datasets: [{{
        label: label,
        data: values,
        backgroundColor: color,
        borderRadius: 4,
      }}]
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        y: {{ beginAtZero: true, max: 1, grid: {{ color: '#313244' }} }},
        x: {{ grid: {{ display: false }} }}
      }}
    }}
  }});
}}

const labels = chartData.map(d => d.name);
const iou = chartData.map(d => d.iou);
const dice = chartData.map(d => d.dice);
const f1 = chartData.map(d => d.f1);
const prec = chartData.map(d => d.precision);
const rec = chartData.map(d => d.recall);

makeBar(document.getElementById('chart_iou'), labels, iou, 'IoU', COLORS);
makeBar(document.getElementById('chart_dice'), labels, dice, 'Dice', COLORS);
makeBar(document.getElementById('chart_f1'), labels, f1, 'F1', COLORS);

new Chart(document.getElementById('chart_pr'), {{
  type: 'bar',
  data: {{
    labels: labels,
    datasets: [
      {{ label: 'Precision', data: prec, backgroundColor: '#89b4fa', borderRadius: 4 }},
      {{ label: 'Recall', data: rec, backgroundColor: '#a6e3a1', borderRadius: 4 }},
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#cdd6f4' }} }} }},
    scales: {{
      y: {{ beginAtZero: true, max: 1, grid: {{ color: '#313244' }} }},
      x: {{ grid: {{ display: false }} }}
    }}
  }}
}});

// ===== Box plots =====
const metricLabelsBox = {{ 'iou': 'IoU', 'dice': 'Dice', 'f1': 'F1', 'precision': 'Precision', 'recall': 'Recall', 'accuracy': 'Accuracy' }};
const boxplotCanvases = ['boxplot_iou', 'boxplot_dice', 'boxplot_f1', 'boxplot_pr'];
const boxplotMetrics = ['iou', 'dice', 'f1', 'precision'];
const boxplotTitles = ['IoU', 'Dice', 'F1', 'Precision + Recall'];

boxplotMetrics.forEach((metric, idx) => {{
  const canvas = document.getElementById('boxplot_' + metric);
  if (!canvas) return;
  const algoNames = Object.keys(boxplotData[metric] || {{}});
  if (algoNames.length === 0) return;

  new Chart(canvas, {{
    type: 'bar',
    data: {{
      labels: algoNames,
      datasets: [{{
        data: algoNames.map(() => 0.5),
        backgroundColor: 'transparent',
        borderColor: 'transparent',
      }}],
      boxData: boxplotData[metric],
    }},
    options: {{
      responsive: true,
      indexAxis: 'y',
      plugins: {{
        legend: {{ display: false }},
        title: {{ display: true, text: boxplotTitles[idx], color: '#cdd6f4' }},
      }},
      scales: {{
        x: {{ beginAtZero: true, max: 1, grid: {{ color: '#313244' }}, title: {{ display: true, text: 'Score', color: '#585b70' }} }},
        y: {{ grid: {{ display: false }} }}
      }}
    }},
    plugins: [boxplotPlugin],
  }});
}});

// PR box plot
const prCanvas = document.getElementById('boxplot_precision_recall');
if (prCanvas && boxplotData['precision']) {{
  const prLabels = [];
  const prBoxData = {{}};
  Object.keys(boxplotData['precision']).forEach(name => {{
    const p = boxplotData['precision'][name];
    const r = boxplotData['recall'][name];
    prLabels.push(name + ' (P)');
    prLabels.push(name + ' (R)');
    prBoxData[name + ' (P)'] = p;
    prBoxData[name + ' (R)'] = r;
  }});
}}

// ===== Scatter plots =====
const scatterMetrics = ['iou', 'dice', 'f1', 'precision'];
scatterMetrics.forEach((metric) => {{
  const canvas = document.getElementById('scatter_' + metric);
  if (!canvas) return;
  const metricData = scatterData[metric];
  if (!metricData) return;
  const algoNames = Object.keys(metricData);
  if (algoNames.length === 0) return;

  const datasets = [];
  algoNames.forEach((name, i) => {{
    const points = metricData[name];
    if (!points || points.length === 0) return;
    datasets.push({{
      label: name,
      data: points.map(p => ({{ x: i, y: p.value }})),
      backgroundColor: COLORS[i % COLORS.length],
      pointRadius: 3,
      pointHoverRadius: 5,
    }});
    datasets.push({{
      label: name + ' outliers',
      data: points.filter(p => p.is_outlier).map(p => ({{ x: i, y: p.value }})),
      backgroundColor: '#f38ba8',
      pointRadius: 5,
      pointStyle: 'triangle',
    }});
  }});

  new Chart(canvas, {{
    type: 'scatter',
    data: {{ datasets }},
    options: {{
      responsive: true,
      plugins: {{
        legend: {{ labels: {{ color: '#cdd6f4', font: {{ size: 10 }} }} }},
        title: {{ display: true, text: metricLabelsBox[metric] + ' — per-sample', color: '#cdd6f4' }},
      }},
      scales: {{
        x: {{
          type: 'category',
          labels: algoNames,
          grid: {{ display: false }},
          ticks: {{ color: '#a6adc8' }}
        }},
        y: {{ beginAtZero: true, max: 1, grid: {{ color: '#313244' }}, title: {{ display: true, text: 'Score', color: '#585b70' }} }}
      }}
    }}
  }});
}});
</script>
</body>
</html>"""


def generate_report(
    all_results: AllResults,
    output_path: str = "test_report.html",
    include_per_snapshot: bool = True,
    comparison: Dict[str, Dict[str, Dict[str, str]]] | None = None,
    include_stats: bool = True,
    include_boxplots: bool = True,
    include_scatter: bool = True,
    include_significance: bool = True,
    include_outliers: bool = True,
    stats_output: str | None = None,
):
    # Если stats отключены — отключаем все расширения
    if not include_stats:
        include_boxplots = False
        include_scatter = False
        include_significance = False
        include_outliers = False

    summary = compute_summary(all_results)

    summary_rows = ""
    stat_headers = ""
    boxplot_sections = ""
    scatter_sections = ""
    significance_sections = ""
    winner_sections = ""
    outlier_section = ""

    if include_stats:
        stat_headers = _stat_header_cells()
        summary_rows = _build_summary_rows(summary)

    if not include_stats:
        # Legacy mode: only mean±std columns for 5 core metrics (no Accuracy)
        legacy_headers = ""
        for label in METRIC_LABELS[:5]:
            legacy_headers += f"<th>{label}</th>"
        stat_headers = legacy_headers
        summary_rows_legacy = ""
        legacy_keys = ["iou", "dice", "f1", "precision", "recall"]
        for entry in summary:
            cells = ""
            for key in legacy_keys:
                cells += f"""<td class="num">{entry.get(f"mean_{key}", 0):.4f} ± {entry.get(f"std_{key}", 0):.4f}</td>"""
            summary_rows_legacy += f"""<tr>
  <td><b>{entry["name"]}</b></td>
  <td>{entry["description"]}</td>
  <td>{entry["num_images"]}</td>
  {cells}
</tr>\n"""
        summary_rows = summary_rows_legacy

    if include_boxplots:
        boxplot_data = _build_boxplot_data(summary)
        boxplot_sections = f"""<div class="chart-container">
  <h3>Distribution (Box-Plot)</h3>
  <div class="boxplot-grid">
    <div class="boxplot-chart"><canvas id="boxplot_iou"></canvas></div>
    <div class="boxplot-chart"><canvas id="boxplot_dice"></canvas></div>
  </div>
  <div class="boxplot-grid">
    <div class="boxplot-chart"><canvas id="boxplot_f1"></canvas></div>
    <div class="boxplot-chart"><canvas id="boxplot_precision_recall"></canvas></div>
  </div>
</div>
"""
    else:
        boxplot_data = "{}"

    if include_scatter:
        scatter_data = _build_scatter_data(all_results)
        scatter_sections = f"""<div class="chart-container">
  <h3>Per-Sample Distribution (Scatter)</h3>
  <div class="scatter-grid">
    <div class="scatter-chart"><canvas id="scatter_iou"></canvas></div>
    <div class="scatter-chart"><canvas id="scatter_dice"></canvas></div>
  </div>
  <div class="scatter-grid">
    <div class="scatter-chart"><canvas id="scatter_f1"></canvas></div>
    <div class="scatter-chart"><canvas id="scatter_precision"></canvas></div>
  </div>
</div>
"""
    else:
        scatter_data = "{}"

    if include_significance:
        wilcoxon_table = compute_wilcoxon_table(all_results)
        significance_sections = _build_significance_section(wilcoxon_table)
        winners = compute_all_winners(all_results)
        winner_sections = _build_winner_table(winners)

    if include_outliers:
        outliers = compute_outlier_table(all_results)
        outlier_section = _build_outlier_table(outliers)

    per_algo_sections = ""
    if include_per_snapshot:
        per_algo_sections = _build_per_algo_sections(all_results)

    comparison_sections = ""
    if comparison:
        comparison_sections = _build_comparison_sections(comparison)

    chart_data_json = _build_chart_data(summary)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = HTML_TEMPLATE.format(
        summary_rows=summary_rows,
        stat_headers=stat_headers,
        per_algo_sections=per_algo_sections,
        comparison_sections=comparison_sections,
        chart_data=chart_data_json,
        boxplot_data=boxplot_data,
        scatter_data=scatter_data,
        boxplot_sections=boxplot_sections,
        scatter_sections=scatter_sections,
        significance_sections=significance_sections,
        winner_sections=winner_sections,
        outlier_section=outlier_section,
        timestamp=timestamp,
    )
    Path(output_path).write_text(html, encoding="utf-8")
    log.info("Report: %s", output_path)

    if stats_output:
        _export_stats_json(summary, all_results, stats_output)


def _export_stats_json(
    summary: List[Dict],
    all_results: AllResults,
    output_path: str,
):
    algo_stats = {}
    for entry in summary:
        algo_name = entry["name"]
        algo_stats[algo_name] = {}
        for key in METRIC_KEYS:
            algo_stats[algo_name][key] = {
                "mean": entry.get(f"mean_{key}", 0),
                "std": entry.get(f"std_{key}", 0),
                "median": entry.get(f"median_{key}", 0),
                "q1": entry.get(f"q1_{key}", 0),
                "q3": entry.get(f"q3_{key}", 0),
                "p5": entry.get(f"p5_{key}", 0),
                "p95": entry.get(f"p95_{key}", 0),
                "min": entry.get(f"min_{key}", 0),
                "max": entry.get(f"max_{key}", 0),
            }
    wilcoxon_table = compute_wilcoxon_table(all_results)
    stats_data = {
        "timestamp": datetime.now().isoformat(),
        "num_algorithms": len(summary),
        "num_samples_per_algo": summary[0]["num_images"] if summary else 0,
        "algorithms": algo_stats,
        "wilcoxon": wilcoxon_table,
    }
    Path(output_path).write_text(
        json.dumps(stats_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("Stats JSON: %s", output_path)
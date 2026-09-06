import argparse
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .baseline import BaselineDataset, run_baseline, export_json
from .classic_algorithms import *  # noqa: F401, F403

log = logging.getLogger(__name__)

EVALUATIONS_DIR = "evaluations"

DEFAULT_CONFIG = {
    "data_root": "test_images",
    "use_cropped": True,
    "algorithms": None,
    "models": [],
    "compare": None,
    "per_snapshot": True,
}


def _load_yaml_or_json(path: Path) -> dict:
    try:
        import yaml
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except ImportError:
        raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def load_config(config_path: str | None, cli_args: dict) -> dict:
    config = dict(DEFAULT_CONFIG)

    if config_path:
        path = Path(config_path)
        if not path.exists():
            log.warning("Config file %s not found, using defaults", config_path)
        else:
            config.update(_load_yaml_or_json(path))

    for key, value in cli_args.items():
        if value is None or value is argparse.SUPPRESS:
            continue
        if key.startswith("no_"):
            config_key = key[3:]
            config[config_key] = not value
        elif key in config or key in ("data_root", "output", "import_root"):
            config[key] = value

    return config


def create_run_dir(base_dir: str = EVALUATIONS_DIR) -> str:
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(base_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id


def get_git_info() -> dict:
    try:
        hash_ = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, cwd=Path(__file__).resolve().parent.parent,
        ).stdout.strip()
        msg = subprocess.run(
            ["git", "log", "-1", "--format=%s"],
            capture_output=True, text=True, check=True, cwd=Path(__file__).resolve().parent.parent,
        ).stdout.strip()
        return {"commit_hash": hash_, "commit_message": msg}
    except (subprocess.CalledProcessError, FileNotFoundError):
        log.warning("Not a git repository or git not available")
        return {"commit_hash": "unknown", "commit_message": "unknown"}


def _generate_eval_html(result: dict, run_id: str) -> str:
    dataset_name = result.get("dataset_name", "unknown")
    timestamp = result.get("timestamp", "")
    rows = ""
    chart_labels = []
    chart_iou = []
    chart_dice = []
    chart_f1 = []
    chart_prec = []
    chart_rec = []

    for algo in result.get("algorithms", []):
        name = algo["name"]
        s = algo.get("source", {})
        rows += f"""<tr>
  <td><b>{name}</b></td>
  <td>{algo.get("description", "")}</td>
  <td class="num">{s.get("num_samples", 0)}</td>
  <td class="num">{s.get("mean_iou", 0):.4f} ± {s.get("std_iou", 0):.4f}</td>
  <td class="num">{s.get("mean_dice", 0):.4f} ± {s.get("std_dice", 0):.4f}</td>
  <td class="num">{s.get("mean_f1", 0):.4f} ± {s.get("std_f1", 0):.4f}</td>
  <td class="num">{s.get("mean_precision", 0):.4f} ± {s.get("std_precision", 0):.4f}</td>
  <td class="num">{s.get("mean_recall", 0):.4f} ± {s.get("std_recall", 0):.4f}</td>
  <td class="num">{s.get("mean_accuracy", 0):.4f} ± {s.get("std_accuracy", 0):.4f}</td>
</tr>\n"""
        chart_labels.append(name)
        chart_iou.append(s.get("mean_iou", 0))
        chart_dice.append(s.get("mean_dice", 0))
        chart_f1.append(s.get("mean_f1", 0))
        chart_prec.append(s.get("mean_precision", 0))
        chart_rec.append(s.get("mean_recall", 0))

    labels_json = json.dumps(chart_labels)
    iou_json = json.dumps(chart_iou)
    dice_json = json.dumps(chart_dice)
    f1_json = json.dumps(chart_f1)
    prec_json = json.dumps(chart_prec)
    rec_json = json.dumps(chart_rec)

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Evaluation Report — {dataset_name}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #11111b; color: #cdd6f4; padding: 2rem; }}
h1 {{ color: #89b4fa; margin-bottom: 0.5rem; }}
h2 {{ color: #89b4fa; margin: 2rem 0 1rem; }}
.meta {{ color: #a6adc8; margin-bottom: 1.5rem; font-size: 0.9rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.85rem; }}
th, td {{ padding: 0.5rem 0.75rem; text-align: left; border-bottom: 1px solid #313244; }}
th {{ background: #181825; color: #a6adc8; font-weight: 600; white-space: nowrap; }}
tr:hover {{ background: #1e1e2e; }}
.num {{ text-align: right; font-family: "JetBrains Mono", "Fira Code", monospace; white-space: nowrap; }}
.chart-container {{ background: #181825; border-radius: 8px; padding: 1.5rem; margin: 1.5rem 0; }}
.chart-row {{ display: flex; gap: 1.5rem; flex-wrap: wrap; }}
.chart-box {{ flex: 1; min-width: 280px; }}
.footer {{ margin-top: 3rem; color: #585b70; font-size: 0.8rem; text-align: center; }}
</style>
</head>
<body>
<h1>Evaluation Report</h1>
<div class="meta">
  Run: <b>{run_id}</b><br>
  Dataset: <b>{dataset_name}</b><br>
  Timestamp: {timestamp}<br>
  Images: {result.get("image_count", {})}
</div>

<div class="chart-container">
  <div class="chart-row">
    <div class="chart-box"><canvas id="chart_iou"></canvas></div>
    <div class="chart-box"><canvas id="chart_dice"></canvas></div>
  </div>
  <div class="chart-row">
    <div class="chart-box"><canvas id="chart_f1"></canvas></div>
    <div class="chart-box"><canvas id="chart_pr"></canvas></div>
  </div>
</div>

<h2>Aggregated Metrics</h2>
<table>
  <thead>
    <tr>
      <th>Algorithm</th>
      <th>Description</th>
      <th>Samples</th>
      <th>IoU</th>
      <th>Dice</th>
      <th>F1</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>

<div class="footer">
Generated by bacterial-colony-area evaluation pipeline &mdash; {run_id}
</div>

<script>
const labels = {labels_json};
const colors = ['#89b4fa', '#a6e3a1', '#f9e2af', '#fab387', '#cba6f7', '#94e2d5'];

function makeBar(ctx, labels, values, label, color) {{
  new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels: labels,
      datasets: [{{
        label: label,
        data: values,
        backgroundColor: colors.slice(0, values.length),
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

makeBar(document.getElementById('chart_iou'), labels, {iou_json}, 'IoU');
makeBar(document.getElementById('chart_dice'), labels, {dice_json}, 'Dice');
makeBar(document.getElementById('chart_f1'), labels, {f1_json}, 'F1');
new Chart(document.getElementById('chart_pr'), {{
  type: 'bar',
  data: {{
    labels: labels,
    datasets: [
      {{ label: 'Precision', data: {prec_json}, backgroundColor: '#89b4fa', borderRadius: 4 }},
      {{ label: 'Recall', data: {rec_json}, backgroundColor: '#a6e3a1', borderRadius: 4 }},
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
</script>
</body>
</html>"""


def run_evaluate(config: dict) -> str:
    run_id = create_run_dir()
    run_dir = Path(EVALUATIONS_DIR) / run_id

    if config.get("import_root"):
        from pathlib import Path as _Path
        from train.dataset_adapters import CocoBboxImporter
        import_root = _Path(config["import_root"])
        data_root = _Path(config["data_root"])
        log.info("Importing dataset from %s to %s", import_root, data_root)
        CocoBboxImporter().build(
            data_root=import_root,
            output_dir=data_root,
            crop=config.get("use_cropped", True),
        )

    log.info("Loading dataset from %s", config["data_root"])
    dataset = BaselineDataset(root=config["data_root"])
    log.info("Dataset: %d source, %d cropped samples", dataset.count_source(), dataset.count_cropped())

    if len(dataset) == 0:
        log.warning("No samples found, aborting")
        return run_id

    result = run_baseline(dataset, use_cropped=config.get("use_cropped", True))

    git_info = get_git_info()
    result["run_id"] = run_id
    result["git_commit"] = git_info["commit_hash"]

    export_json(result, str(run_dir / "report.json"))

    html = _generate_eval_html(result, run_id)
    (run_dir / "report.html").write_text(html, encoding="utf-8")

    try:
        import yaml as _yaml
        config_copy = Path(run_dir / "config.yaml")
        config_copy.write_text(_yaml.safe_dump(config, allow_unicode=True, default_flow_style=False), encoding="utf-8")
    except ImportError:
        config_copy = Path(run_dir / "config.json")
        config_copy.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    run_info = {
        "run_id": run_id,
        "timestamp": result.get("timestamp", ""),
        "dataset_path": config["data_root"],
        "dataset_size": {"source": dataset.count_source(), "cropped": dataset.count_cropped()},
        "algorithms_run": [a["name"] for a in result.get("algorithms", [])],
        "git_commit": git_info["commit_hash"],
        "git_message": git_info["commit_message"],
    }
    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("Evaluation complete: %s", run_dir)
    return run_id
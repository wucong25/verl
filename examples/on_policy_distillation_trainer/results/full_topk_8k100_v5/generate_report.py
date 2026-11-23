# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Render saved scalar metrics locally; never import or run training code.

From this directory: uv run --no-project --with matplotlib python generate_report.py
"""

import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
ACC = "val-core/hiyouga/geometry3k/acc/mean@1"
MODES = ("full", "topk")
COLORS = {"full": "#2563eb", "topk": "#e17623"}
LABELS = {"full": "Full", "topk": "Top-k (k=64)"}


def read_metrics(mode):
    data = (ROOT / f"{mode}.metrics.txt").read_bytes()
    expected = json.loads((ROOT / "provenance.json").read_text())["runs"][mode]["metrics_sha256"]
    assert hashlib.sha256(data).hexdigest() == expected, f"Source hash mismatch: {mode}"
    rows = {}
    for line in data.decode().splitlines():
        match = re.fullmatch(r"step:(\d+) - (.+)", line)
        assert match, f"Invalid metric line: {mode}"
        step = int(match[1])
        row = rows.setdefault(step, {})
        for field in match[2].split(" - "):
            key, value = field.rsplit(":", 1)
            number = float(value)
            assert math.isfinite(number), f"Non-finite metric: {mode}/{step}/{key}"
            assert key not in row or row[key] == number, f"Conflicting metric: {mode}/{step}/{key}"
            row[key] = number
    assert sorted(rows) == list(range(max(rows) + 1)), f"Missing step: {mode}"
    for step, row in rows.items():
        if step:
            assert row["training/global_step"] == step
            assert "actor/distillation/loss" in row and "timing_s/step" in row
        if ACC in row:
            assert 0 <= row[ACC] <= 1 and (row[ACC] * 64).is_integer()
    return rows


def describe(values):
    return {
        "count": len(values),
        "first": values[0],
        "last": values[-1],
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "population_std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def training_stats(rows, end):
    keys = sorted(set.intersection(*(set(rows[s]) for s in range(1, end + 1))))
    # Validation metrics occur only at checkpoints and are summarized separately.
    keys = [key for key in keys if not key.startswith("val-")]
    return {key: describe([rows[s][key] for s in range(1, end + 1)]) for key in keys}


def main():
    runs = {mode: read_metrics(mode) for mode in MODES}
    common_end = min(max(rows) for rows in runs.values())
    summary = {
        "attempt": "v5",
        "planned_steps_per_run": 100,
        "common_training_window": [1, common_end],
        "both_runs_complete": all(max(rows) == 100 for rows in runs.values()),
        "statistics_definition": "Unweighted per-step statistics; population standard deviation.",
        "runs": {},
    }
    accuracy_rows = []
    for mode, rows in runs.items():
        validation = [
            {"step": s, "accuracy": r[ACC], "correct": int(r[ACC] * 64), "total": 64}
            for s, r in sorted(rows.items())
            if ACC in r
        ]
        for record in validation:
            accuracy_rows.append({"mode": mode, **record})
        summary["runs"][mode] = {
            "completed_training_steps": max(rows),
            "planned_steps_complete": max(rows) == 100,
            "baseline_validation": validation[0],
            "latest_validation": validation[-1],
            "validation": validation,
            "recorded_step_time_sum_seconds": sum(r.get("timing_s/step", 0) for r in rows.values()),
            "all_completed_steps": training_stats(rows, max(rows)),
            "common_window": training_stats(rows, common_end),
        }
    summary["baseline_accuracy_matches"] = runs["full"][0][ACC] == runs["topk"][0][ACC]
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    keys = sorted({k for rows in runs.values() for row in rows.values() for k in row})
    with (ROOT / "metrics.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=["mode", "step", *keys])
        writer.writeheader()
        for mode, rows in runs.items():
            for step, row in sorted(rows.items()):
                writer.writerow({"mode": mode, "step": step, **row})
    with (ROOT / "validation.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=["mode", "step", "accuracy", "correct", "total"])
        writer.writeheader()
        writer.writerows(accuracy_rows)
    with (ROOT / "statistics.csv").open("w", newline="") as out:
        columns = ["mode", "window", "metric", *describe([1]).keys()]
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        for mode in MODES:
            for window in ("all_completed_steps", "common_window"):
                for metric, stats in summary["runs"][mode][window].items():
                    writer.writerow({"mode": mode, "window": window, "metric": metric, **stats})

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "svg.hashsalt": "full-topk-v5"}
    )
    panels = [
        (ACC, "Validation answer accuracy (64 examples)", "Accuracy (%)", MODES, 100),
        ("actor/distillation/loss", "Full-vocabulary KL loss", "Full KL loss", ("full",), 1),
        ("actor/distillation/loss", "Truncated top-k loss (k=64)", "Top-k loss", ("topk",), 1),
        ("response_length/mean", "Training response length", "Mean tokens", MODES, 1),
        ("response_length/clip_ratio", "Training response cap ratio", "Ratio", MODES, 1),
        ("actor/grad_norm", "Actor gradient norm", "Norm", MODES, 1),
        ("timing_s/step", "Recorded training step time", "Seconds", MODES, 1),
        ("actor/perf/max_memory_allocated_gb", "Actor peak allocated memory", "GiB", MODES, 1),
        ("actor/entropy", "Actor entropy", "Entropy", MODES, 1),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(17, 12), constrained_layout=True)
    for ax, (key, title, ylabel, modes, scale) in zip(axes.flat, panels, strict=True):
        for mode in modes:
            points = [(s, r[key] * scale) for s, r in sorted(runs[mode].items()) if key in r]
            ax.plot(
                *zip(*points, strict=True),
                color=COLORS[mode],
                label=LABELS[mode],
                linewidth=1.4,
                marker="o" if key == ACC else None,
                markersize=3,
            )
        ax.axvline(common_end, color="#888888", linestyle=":", linewidth=0.9)
        ax.set(title=title, xlabel="Training step", ylabel=ylabel, xlim=(0, 100))
        if key == ACC:
            ax.set_ylim(0, 100)
        if key == "response_length/clip_ratio":
            ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.2)
        ax.legend(loc="best", fontsize=9)
    fig.suptitle(
        f"Full vs Top-k | v5 | seed 42 | response cap 8192 | learning rate 1e-7\n"
        f"Saved metrics: Full 100/100, Top-k {max(runs['topk'])}/100 (incomplete). "
        "Dotted line: common-window endpoint.\n"
        "Raw per-step curves; loss definitions differ. No extrapolation or new training.",
        fontsize=14,
    )
    fig.savefig(ROOT / "training_curves.png", dpi=160)
    fig.savefig(ROOT / "training_curves.svg", metadata={"Date": None})
    svg = ROOT / "training_curves.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    print(
        json.dumps(
            {
                mode: {
                    key: summary["runs"][mode][key]
                    for key in (
                        "completed_training_steps",
                        "baseline_validation",
                        "latest_validation",
                        "recorded_step_time_sum_seconds",
                    )
                }
                for mode in MODES
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

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
"""Summarize complete OPD console logs without treating reward as accuracy."""

import argparse
import json
import math
import re
import statistics
from pathlib import Path

ACCURACY = "val-core/hiyouga/geometry3k/acc/mean@1"
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
METRICS = {
    "step_seconds": "timing_s/step",
    "actor_update_seconds": "timing_s/update_actor",
    "actor_peak_allocated_gib": "actor/perf/max_memory_allocated_gb",
    "actor_peak_reserved_gib": "actor/perf/max_memory_reserved_gb",
    "response_clip_ratio": "response_length/clip_ratio",
}


def parse_log(path):
    rows = {}
    for line in ANSI.sub("", Path(path).read_text(errors="replace")).splitlines():
        match = re.search(r"(?:^|\s)step:(\d+) - ", line)
        if match is None:
            continue
        row = rows.setdefault(int(match[1]), {})
        for field in line[match.end() :].split(" - "):
            key, sep, value = field.partition(":")
            if not sep:
                continue
            try:
                number = float(value)
            except ValueError:
                continue
            if not math.isfinite(number):
                raise ValueError(f"Non-finite metric at step {match[1]}: {key}")
            if key in row and row[key] != number:
                raise ValueError(f"Conflicting metric at step {match[1]}: {key}")
            row[key] = number
    return rows


def summarize(rows, steps):
    if steps < 1:
        raise ValueError("steps must be positive")
    if set(rows) != set(range(steps + 1)):
        raise ValueError(f"Expected steps 0..{steps}, got {sorted(rows)}")
    for step in (0, steps):
        if ACCURACY not in rows[step] or not 0 <= rows[step][ACCURACY] <= 1:
            raise ValueError(f"Missing or invalid answer accuracy at step {step}")
    result = {"steps": steps, "accuracy": {s: r[ACCURACY] for s, r in rows.items() if ACCURACY in r}}
    for label, key in METRICS.items():
        try:
            values = [rows[step][key] for step in range(1, steps + 1)]
        except KeyError as exc:
            raise ValueError(f"Missing required training metric: {key}") from exc
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError(f"Invalid training metric: {key}")
        result[label] = {"mean": statistics.mean(values), "max": max(values)}
    return result


def compare(full, topk, steps):
    result = {"full": summarize(full, steps), "topk": summarize(topk, steps)}
    result["baseline_accuracy_matches"] = full[0][ACCURACY] == topk[0][ACCURACY]
    result["limitations"] = [
        "Single-seed pilot; matching aggregate accuracy does not prove identical baseline outputs.",
        "Memory covers actor allocator metrics, not teacher, rollout or whole-device peak usage.",
        "Step timings include rollout and communication; different response lengths confound speed comparisons.",
        "Full KL and truncated top-k losses are not numerically comparable.",
    ]
    if not result["baseline_accuracy_matches"]:
        result["limitations"].append(
            "Baseline accuracy differs; do not attribute quality differences to the loss mode."
        )
    if any(result[mode]["response_clip_ratio"]["mean"] > 0.5 for mode in ("full", "topk")):
        result["limitations"].append("Most training responses reach the token cap; inspect saved generations.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--topk", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(parse_log(args.full), parse_log(args.topk), args.steps)
    result["logs"] = {"full": str(args.full), "topk": str(args.topk)}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

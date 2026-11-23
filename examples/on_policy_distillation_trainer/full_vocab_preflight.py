# Copyright 2026 verl contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Read-only preflight for aligned full-vocabulary OPD model pairs."""

import argparse
import importlib.util
import json
from pathlib import Path


def inspect_model(path):
    root = Path(path)
    issues = []
    config_path = root / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    if not config:
        issues.append("Missing config.json")
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        shards = set(json.loads(index_path.read_text())["weight_map"].values())
        missing = sorted(name for name in shards if not (root / name).is_file())
        if missing:
            issues.append(f"Missing weight shards: {missing}")
    elif not (root / "model.safetensors").is_file():
        issues.append("Missing model.safetensors or model.safetensors.index.json")
    tokenizer_path = root / "tokenizer.json"
    tokenizer = json.loads(tokenizer_path.read_text()) if tokenizer_path.exists() else None
    if tokenizer is None:
        issues.append("Missing tokenizer.json; token ID compatibility is unverified")
    return {
        "path": str(root),
        "vocab_size": config.get("text_config", config).get("vocab_size"),
        "issues": issues,
    }, tokenizer


def compare_models(teacher_path, student_path):
    teacher, teacher_tokenizer = inspect_model(teacher_path)
    student, student_tokenizer = inspect_model(student_path)
    issues = []
    if teacher["vocab_size"] != student["vocab_size"]:
        issues.append("Different output vocabulary sizes; direct full-vocabulary KL is undefined")
    if teacher_tokenizer is not None and student_tokenizer is not None:
        for key in ("model", "added_tokens", "normalizer", "pre_tokenizer", "decoder"):
            if teacher_tokenizer.get(key) != student_tokenizer.get(key):
                issues.append(f"Different tokenizer {key}; explicit alignment validation required")
    return {
        "teacher": teacher,
        "student": student,
        "compatibility_issues": issues,
        "ready": not (teacher["issues"] or student["issues"] or issues),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--student", required=True)
    args = parser.parse_args()
    report = compare_models(args.teacher, args.student)
    report["installed_backends"] = {
        name: importlib.util.find_spec(name) is not None for name in ("torch", "torch_npu", "megatron", "fsdp_turbo")
    }
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ready"] else 1)


if __name__ == "__main__":
    main()

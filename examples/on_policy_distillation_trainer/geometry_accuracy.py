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
"""Keep the geometry reward unchanged and report answer accuracy separately."""

from verl.utils.reward_score.geo3k import acc_reward, format_reward
from verl.utils.reward_score.geo3k import compute_score as geometry_score


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source != "hiyouga/geometry3k":
        raise ValueError(f"This comparison expects geometry3k, received {data_source!r}")
    return {
        "score": geometry_score(solution_str, ground_truth),
        "acc": acc_reward(solution_str, ground_truth),
        "format": format_reward(solution_str),
    }

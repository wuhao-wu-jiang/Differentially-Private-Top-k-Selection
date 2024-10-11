# coding=utf-8
# Original Work (dp_topk):
#     This file is the same as the file "differential_privacy.py" from the
#     open-source library dp_topk, available at (as of May 2024):
#
#     https://github.com/google-research/google-research/tree/master/dp_topk
#
# Original Authors:
#     Google LLC
#     Jennifer Gillenwater
#     Matthew Joseph
#     Andrés Muñoz Medina
#     Mónica Ribero
#
# Copyright 2024 The Google Research Authors.
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
# -----------------------------------------------------------------------------

"""Class for switching between add/remove and swap differential privacy."""

import enum

class NeighborType(enum.Enum):
    ADD_REMOVE = 1
    SWAP = 2


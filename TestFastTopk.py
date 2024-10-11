# coding=utf-8
# Copyright 2024 Hao Wu, Hanwen Zhang.
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

import math
import operator

import numpy as np
from absl.testing import absltest

import FastTopk
from DP_Parameters import NeighborType
from FastTopk import fast_joint_sampling_dp_top_k



def assert_array_less_equal(x, y, err_msg='', verbose=True):
    return np.testing.assert_array_compare(
        operator.__le__,
        x,
        y,
        err_msg=err_msg,
        verbose=verbose,
        header='x is not less than or equal to y.',
        equal_inf=False)


def compute_confidence_interval_binary_rv(rv_probability, num_trial, interval_probability):
    log_one_over_interval_probability = np.log(1.0 / interval_probability)
    interval = (np.sqrt(2 * rv_probability * (
            1 - rv_probability) * num_trial * log_one_over_interval_probability)
                + 2.0 / 3 * log_one_over_interval_probability)
    return interval


class JointTest(absltest.TestCase):

    def test_sort_largest(self):
        bound = 10000
        hist = np.repeat(np.arange(bound), 5)
        k = 1
        k_largest = FastTopk.sort_k_largest(hist, k)
        expected_largest = bound - 1
        self.assertEqual(k_largest[0], expected_largest)

    def test_sort_k_largest(self):
        bound = 100000
        hist = np.arange(bound)
        k = 5000
        k_largest = FastTopk.sort_k_largest(hist, k)
        expected_largest = np.arange(bound - k, bound)[::-1]
        np.testing.assert_array_equal(k_largest, expected_largest)

    def test_construct_index_groups(self):
        hist = np.repeat(np.arange(100), 70)
        index_groups = FastTopk.construct_index_groups(hist)
        assert len(index_groups) == 100
        distinct_count = len(np.unique(np.array([pairs[0] for pairs in index_groups])))
        assert distinct_count == 100
        for val, pairs in enumerate(index_groups):
            np.testing.assert_array_equal(np.arange(val * 70, val * 70 + 70), np.sort(np.array(pairs[1])))

    def test_sort_group_indices_by_score_above_threshold(self):
        # hist = [3, 4, 3, 2, 4, 6, 7, 9]
        index_groups = [(3, [0, 2]), (4, [1, 4]), (2, [3]), (6, [5]), (7, [6]), (9, [7])]
        threshold = 5
        sorted_index_groups = FastTopk.sort_group_indices_by_score_above_threshold(index_groups, threshold)
        expected = [(9, [7]), (7, [6]), (6, [5])]
        for pair1, pair2 in zip(sorted_index_groups, expected):
            assert pair1[0] == pair2[0] and pair1[1][0] == pair2[1][0]

    def test_initialize_counter_indices(self):
        k = 3
        hist = np.array([5, 5, 5, 8, 8])
        d = len(hist)
        true_top_k_scores = np.array([8, 8, 5])
        sorted_profile = [(8, 2), (5, 3)]
        counter_positions = FastTopk.initialize_counter_indices(sorted_profile, k)
        for i, pos in enumerate(counter_positions):
            assert true_top_k_scores[i] == sorted_profile[pos][0]

    def test_report_noisy_max_with_grumbel_noises(self):
        log_terms = np.arange(0, 10, 0.5)
        sampled_counts = np.zeros(len(log_terms))
        num_trials = 10000
        for _ in range(num_trials):
            sampled_counts[FastTopk.report_noisy_max_with_grumbel_noises(log_terms)] += 1
        terms = np.exp(log_terms)
        expected_sample_probs = terms / np.sum(terms)
        expected_counts = expected_sample_probs * num_trials
        interval_probability = 1.0 / 30 / expected_sample_probs.size
        expected_sample_widths = compute_confidence_interval_binary_rv(expected_sample_probs, num_trials,
                                                                       interval_probability)
        np.testing.assert_array_less(sampled_counts, expected_counts + expected_sample_widths)
        np.testing.assert_array_less(expected_counts - expected_sample_widths, sampled_counts)


    def test_brute_compute_log_sequence_count_matrix(self):
        hist = np.array([1, 5, 7, 9])
        k = 2
        d = len(hist)
        true_top_k_scores = [9, 7]
        tau = 3
        with np.errstate(divide='ignore'):
            # the error-sequence matrix in this case is given by
            # [
            #   [{(9, 7)}, {}],
            #   [{}, {}],
            #   [{(7, 9), (7, 5)}, {(9,5)}],
            #   [{(5, 9), (5, 7), (5, 1), (1, 9), (1, 7), (1, 5)}, {(9, 1), (7, 1)}]
            # ]
            expected_log_sequence_count_matrix = np.array([[1, 0], [0, 0], [2, 1], [6, 2]])
        brute_log_sequence_count_matrix = FastTopk.brute_compute_log_sequence_count_matrix(hist, true_top_k_scores,
                                                                                           tau, d, k)
        np.testing.assert_array_equal(np.exp(brute_log_sequence_count_matrix),
                                      expected_log_sequence_count_matrix)

    def test_compute_log_sequence_count_matrix(self):
        # counter = 0
        # np.random.seed(0)
        for d in [5, 6, 7]:
            for k in [2, 3, 4]:
                for _ in range(100):
                    hist = np.sort(np.random.choice(10 * d, size=d))[::-1]
                    true_top_k_scores = FastTopk.sort_k_largest(hist, k)
                    tau = 5
                    lowest_score_to_be_considered = true_top_k_scores[-1] - tau
                    index_groups = FastTopk.construct_index_groups(hist)
                    sorted_indices_groups = FastTopk.sort_group_indices_by_score_above_threshold(index_groups,
                                                                                                 lowest_score_to_be_considered)

                    log_sequence_count_matrix = FastTopk.compute_log_sequence_count_matrix(true_top_k_scores,
                                                                                           sorted_indices_groups,
                                                                                           tau, d, k)
                    brute_log_sequence_count_matrix = FastTopk.brute_compute_log_sequence_count_matrix(hist,
                                                                                                       true_top_k_scores,
                                                                                                       tau, d,
                                                                                                       k)
                    np.testing.assert_array_almost_equal(
                        log_sequence_count_matrix, brute_log_sequence_count_matrix, decimal=6)

        ###########
        hist = np.arange(25)
        d = len(hist)
        k = 2
        true_top_k_scores = np.array([24, 23])
        epsilon = 1
        failure_probability = 1.0 / 10
        tau = math.ceil(1.0 / epsilon * np.log(d * (d - 1) / failure_probability))
        lowest_score_to_be_considered = true_top_k_scores[-1] - tau
        index_groups = FastTopk.construct_index_groups(hist)
        sorted_indices_groups = FastTopk.sort_group_indices_by_score_above_threshold(index_groups,
                                                                                     lowest_score_to_be_considered)
        sorted_profile = np.array([(key, len(indices)) for key, indices in sorted_indices_groups])
        log_sequence_count_matrix = FastTopk.compute_log_sequence_count_matrix(true_top_k_scores, sorted_indices_groups,
                                                                               tau, d, k)
        brute_log_sequence_count_matrix = FastTopk.brute_compute_log_sequence_count_matrix(hist,
                                                                                           true_top_k_scores,
                                                                                           tau, d,
                                                                                           k)
        np.testing.assert_array_almost_equal(
            log_sequence_count_matrix, brute_log_sequence_count_matrix, decimal=6)

    # def test_compute_error_truncation_threshold(self):

    def helper_test_error_idx(self, neighbor_type):
        search_range = 3
        error_counts_matrix = np.array([[8, 0], [1, 2], [3, 4], [5, 6]])
        with np.errstate(divide='ignore'):
            log_error_counts_matrix = np.log(error_counts_matrix)
        epsilon = 2.5

        if neighbor_type is NeighborType.SWAP:
            scaling = 4
        else:
            scaling = 2

        # Explicitly compute expected distribution by exponentiating rather than
        # calling racing_sample as sample_diff_idx does.
        expected_sample_probs = error_counts_matrix * (np.exp(-(epsilon / scaling) * np.arange(4))[:, np.newaxis])
        expected_sample_probs_norm = np.sum(expected_sample_probs)
        expected_sample_probs /= expected_sample_probs_norm
        # Set a relatively tight width since we are comparing with the exact
        # probabilities of the expected distribution.
        sample_width_scaling_factor = 1

        num_trials = 100000
        sampled_counts = np.zeros(error_counts_matrix.shape)
        for _ in range(num_trials):
            sampled_counts[
                FastTopk.sample_error_idx(log_error_counts_matrix, search_range, epsilon, neighbor_type)] += 1

        interval_probability = 1.0 / 30 / expected_sample_probs.size
        expected_counts = expected_sample_probs * num_trials
        expected_sample_widths = compute_confidence_interval_binary_rv(expected_sample_probs, num_trials,
                                                                       interval_probability)
        np.testing.assert_array_less(sampled_counts, expected_counts + expected_sample_widths)
        np.testing.assert_array_less(expected_counts - expected_sample_widths, sampled_counts)


    def test_sample_error_idx_add_remove(self):
        self.helper_test_error_idx(NeighborType.ADD_REMOVE)

    def test_sample_error_idx_swap(self):
        self.helper_test_error_idx(NeighborType.SWAP)

    def test_sample_swap_to_the_back(self):
        hist = np.arange(100)
        num_trials = 10000
        freq = np.zeros(hist.shape)
        for _ in range(num_trials):
            FastTopk.sample_swap_to_the_back(hist)
            freq[hist[-1]] += 1

        expected_freq = np.full(shape=hist.shape, fill_value=100)
        expected_probs = len(hist) / num_trials
        interval_probability = 1.0 / 30 / expected_freq.size
        expected_sample_widths = compute_confidence_interval_binary_rv(expected_probs, num_trials,
                                                                       interval_probability)
        np.testing.assert_array_less(freq, expected_freq + expected_sample_widths)
        np.testing.assert_array_less(expected_freq - expected_sample_widths, freq)



    def test_sample_sequence(self):
        # hist = [5, 5, 5, 0, 1, 2, 3, 4]
        d = 8
        k = 2
        tau = 2
        true_top_k_scores = np.array([5, 5])
        sorted_indices_groups = [(5, [0, 1, 2]), (4, [4])]
        error = 1
        error_col = 1
        sequence_counts = np.zeros(4)
        num_trials = 10000
        for _ in range(num_trials):
            sequence = FastTopk.sample_sequence(sorted_indices_groups, d, k, true_top_k_scores, tau, error, error_col)
            if sequence[0] not in [0, 1, 2] or sequence[1] != 4:
                sequence_counts[3] += 1
            else:
                sequence_counts[sequence[0]] += 1
        expected_probs = np.array([1. / 3, 1. / 3, 1. / 3, 0])
        interval_probability = 1.0 / 30 / expected_probs.size
        expected_counts = expected_probs * num_trials
        expected_sample_widths = compute_confidence_interval_binary_rv(expected_probs, num_trials,
                                                                       interval_probability)
        assert_array_less_equal(sequence_counts, expected_counts + expected_sample_widths)
        assert_array_less_equal(expected_counts - expected_sample_widths, sequence_counts)


    def helper_test_fast_joint_sampling_dp_top_k(self, neighbor_type):
        hist = np.arange(25)
        true_top_k_scores = np.array([24, 23])
        d = len(hist)
        k = 2
        epsilon = 1
        failure_probability = 1.0 / 100
        if neighbor_type is NeighborType.SWAP:
            sensitivity = 2
        else:
            sensitivity = 1

        tau = FastTopk.compute_error_truncation_threshold(epsilon=epsilon, d=d, k=k, failure_probability=failure_probability, neighbor_type=neighbor_type)
        sequence_counts = np.zeros((d, d))
        num_trials = 10000
        for i in range(num_trials):
            sequence = fast_joint_sampling_dp_top_k(item_counts=hist, k=k, epsilon=epsilon, failure_probability=failure_probability, neighbor_type=neighbor_type)
            sequence_counts[sequence[0], sequence[1]] += 1
            assert sequence[0] != sequence[1], f"{i} {sequence[0]} {sequence[1]}"

        for i in range(d):
            assert sequence_counts[i, i] == 0

        sampling_weights = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                if j == i:
                    sampling_weights[i, j] = -np.inf
                else:
                    weight = np.max([true_top_k_scores[0] - i, true_top_k_scores[1] - j])
                    weight = min(weight, tau)
                    sampling_weights[i, j] = -weight

        sampling_weights = np.exp(sampling_weights * epsilon / (2 * sensitivity))
        expected_count_probs = sampling_weights / np.sum(sampling_weights)

        interval_probability = 1.0 / 30 / expected_count_probs.size
        expected_counts = expected_count_probs * num_trials
        expected_count_widths = compute_confidence_interval_binary_rv(expected_count_probs, num_trials,
                                                                      interval_probability)
        assert_array_less_equal(sequence_counts, expected_counts + expected_count_widths)
        assert_array_less_equal(expected_counts - expected_count_widths, sequence_counts)


    def test_fast_joint_sampling_dp_top_k_add_remove(self):
        # np.random.seed(4)
        self.helper_test_fast_joint_sampling_dp_top_k(NeighborType.ADD_REMOVE)

    def test_fast_joint_sampling_dp_top_k_swap(self):
        self.helper_test_fast_joint_sampling_dp_top_k(NeighborType.SWAP)


if __name__ == '__main__':
    absltest.main()

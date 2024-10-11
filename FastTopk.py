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



import itertools
import math
from collections import defaultdict

import numpy as np

from DP_Parameters import NeighborType


def sort_k_largest(hist, k):
    """
    Returns:
         true k-largest scores in decreasing order

    Args:
      hist: Array of integers.
      k: An integer indicating the number of desired items.
    """
    # Use np.partition to partition the array around the k-th largest element
    partitioned = np.partition(hist, -k)

    # Sort the k largest elements
    sorted_k_largest = np.sort(partitioned[-k:])[::-1]

    return sorted_k_largest


# This function can be vectorized.
# A potential future implementation might leverage the pandas library and its groupby function.

def construct_index_groups(hist):
    """Returns a list, s.t. each element is a tuple (val, [idx0, idx1, idx2, ...]), containing a value (val),
    and a list of indices (idx0, idx1, idx2, ...) the corresponding values of which in hist equal (val)

    Args:
      hist: Array of integers.
    """
    grouped_indices = defaultdict(list)
    for idx, val in enumerate(hist):
        grouped_indices[val].append(idx)
    return list(grouped_indices.items())


def sort_group_indices_by_score_above_threshold(index_groups, threshold):
    """
    Returns:
        a sorted list of tuples whose first coordinate is greater than the (threshold), in decreasing order, also
        according to the first coordinate. Each element is a tuple (val, [idx0, idx1, idx2, ...]),
        containing a value (val), and a list of indices (idx0, idx1, idx2, ...).

    Args:
        threshold: a lower bound on the first coordinate of the tuples

        index_groups: a list, s.t. each element is a tuple (val, [idx0, idx1, idx2, ...]), containing a value (val),
            and a list of indices (idx0, idx1, idx2, ...)
    """
    # Filter the elements that are not smaller than the threshold
    above_threshold = [pair for pair in index_groups if pair[0] > threshold]

    # Sort the indices of the array based on the first coordinate
    above_threshold.sort(key=lambda pair: pair[0], reverse=True)

    return above_threshold


def initialize_counter_indices(sorted_profile, k):
    """
    Returns:
        A array of indices (i_1, i_2, ..., i_k), s.t., sorted_profile[i_j][0] equals hist[j], for j = 1, 2, ..., k
    Args:
        sorted_profile: a sorted list of tuples in decreasing order, according to the first coordinate.
        Each tuple has the form (t, num), num = # positions in (hist) that have value t, where (hist) is the input to
        the function (fast_joint_sampling_dp_top_k)
        k: An integer indicating the number of desired items.
    """
    lower_bound_indices = []
    for index, (score, freq) in enumerate(sorted_profile):
        if len(lower_bound_indices) < k:
            # Calculate how many times to repeat the current index
            repeats = min(freq, k - len(lower_bound_indices))
            # Append the current index to lower_bound_indices multiple times
            lower_bound_indices.extend([index] * repeats)
        else:
            break
    return np.array(lower_bound_indices)


def brute_compute_log_sequence_count_matrix(hist, true_top_k_scores, tau, d, k):
    """Computes a matrix of log(the number of sequence count) by checking all possible sequences

        Args:
          hist: Array of integers.
          d: An integer indicating the total number of items.
          true_top_k_scores: An array of the k largest scores, in decreasing order.
          k: An integer indicating the number of desired items.
          tau: A pruning threshold, as defined in our paper.

        Returns:
          A matrix log_sequence_count_matrix where, log_sequence_count_matrix[i][j] =
          log(# of sequences whose maximum error (truncated at the value of tau) equals i,
            and the first coordinate the maximum error occurs is given by j).
    """
    possible_sequences = itertools.permutations(np.arange(d), k)
    log_sequence_count_matrix = np.zeros((tau + 1, k))
    for sequence in possible_sequences:
        # A clipping step: if the score exceed some threshold tau, we set it to tau
        diffs = np.minimum(true_top_k_scores - hist[np.array(sequence)], tau)
        # Compute the first position that incurs the maximum error
        score_index = np.argmax(diffs)
        # Retrieve the maximum error
        score = diffs[score_index]
        log_sequence_count_matrix[score][score_index] += 1
    # Ignore warnings from taking log(0). This produces -np.inf as intended.
    with np.errstate(divide="ignore"):
        return np.log(log_sequence_count_matrix)


def compute_log_sequence_count_matrix(true_top_k_scores, sorted_indices_groups, tau, d, k):
    """Computes a matrix of log(the number of sequence count) by the method proposed in our paper

    Args:
      sorted_indices_groups:  a sorted list of tuples, in decreasing order, according to the first coordinate.
        Each element in the list is a tuple (val, [idx0, idx1, idx2, ...]), containing a value (val),
        and a list of indices (idx0, idx1, idx2, ...).
      d: An integer indicating the total number of items.
      true_top_k_scores: An array of the k largest scores, in decreasing order.
      k: An integer indicating the number of desired items.
      tau: A pruning threshold, as defined in our paper.

    Returns:
      A matrix log_sequence_count_matrix where, log_sequence_count_matrix[i][j] =
      log(# of sequences whose maximum error (truncated at the value of tau) equals i,
        and the first coordinate the maximum error occurs is given by j).
    """

    # start_time = time.time()
    list_sorted_profile = [(key, len(indices)) for key, indices in sorted_indices_groups]
    sorted_profile = np.array(list_sorted_profile)
    sorted_profile_prefix_sum = np.cumsum(sorted_profile[:, 1])
    counter_positions = initialize_counter_indices(sorted_profile, k)
    # Initialize log_sequence_count_matrix to zeros
    log_sequence_count_matrix = np.zeros((tau + 1, k))

    # we deliberately create a dummy item in the sorted_profile for convenience
    list_sorted_profile.append((-np.inf, 0))
    sorted_profile = np.array(list_sorted_profile)

    # code for handling sequence with maximum error = 0
    # observe that, if a sequence has maximum error 0, it must happen at the first position of the sequence
    current_log_counters = np.log(
        sorted_profile_prefix_sum[counter_positions] - np.arange(k))
    previous_prefix_sums = np.cumsum(current_log_counters)
    log_sequence_count_matrix[0][:] = -np.inf
    log_sequence_count_matrix[0][0] = np.sum(current_log_counters)

    # end_time = time.time()
    # print("Initialization Time " + str(end_time - start_time))

    # code for handling sequence with maximum error = 1 ... tau - 1
    for error in range(1, tau):
        # Find the positions that meet the condition
        condition = (sorted_profile[counter_positions + 1, 0] >= true_top_k_scores - error)

        # Update counter_positions and counter_diffs based on the condition
        counter_positions += condition
        # counter_diffs[j] = # sequences whose maximum error is tau, and the first position this error happens is j
        counter_diffs = condition * sorted_profile[counter_positions, 1]

        current_log_counters = np.log(
            sorted_profile_prefix_sum[counter_positions] - np.arange(k))
        current_suffix_sums = np.cumsum(current_log_counters[::-1])[::-1]

        # We want need to compute the following:
        # log_sequence_count_matrix[error][j]
        #   = previous_prefix_sums[j - 1]
        #   + np.log(counter_diffs)[j]
        #   + current_suffix_sums[j + 1]

        # Ignore warnings from taking log(0). This produces -np.inf as intended.
        with np.errstate(divide="ignore"):
            log_sequence_count_matrix[error][:] = np.log(counter_diffs)
        log_sequence_count_matrix[error][:-1] += current_suffix_sums[1:]
        log_sequence_count_matrix[error][1:] += previous_prefix_sums[:-1]

        previous_prefix_sums = np.cumsum(current_log_counters)

    # code for handling sequence with maximum error >= tau
    current_log_counters = np.log(d - np.arange(k))
    current_suffix_sums = np.cumsum(current_log_counters[::-1])[::-1]
    # we should have sorted_profile[counter_positions, 0] >= true_top_k_scores - (tau - 1)
    # and sorted_profile[counter_positions, 0] < true_top_k_scores - tau
    counter_diffs = d - sorted_profile_prefix_sum[counter_positions]
    # Ignore warnings from taking log(0). This produces -np.inf as intended.
    with np.errstate(divide="ignore"):
        log_sequence_count_matrix[tau][:] = np.log(counter_diffs)
    log_sequence_count_matrix[tau][:-1] += current_suffix_sums[1:]
    log_sequence_count_matrix[tau][1:] += previous_prefix_sums[:-1]

    return log_sequence_count_matrix


def compute_error_truncation_threshold(epsilon, d, k, failure_probability, neighbor_type):
    """
    Args:
        neighbor_type: neighboring dataset type
        failure_probability: failure probability for sampling a sequence
            whose error score no less than the computed truncation threshold
        k: An integer indicating the number of desired items.
        d: An integer indicating the total number of items.
        epsilon: privacy parameter

    Returns:
        the truncation threshold defined in the paper
    """
    if neighbor_type is NeighborType.SWAP:
        sensitivity = 2
    else:
        sensitivity = 1

    err_range = (2 * sensitivity) * 1.0 / epsilon * (
            np.sum(np.log(np.arange(d - k + 1, d + 1)))
            + np.log(1.0 / failure_probability)
    )
    return math.ceil(err_range)


def report_noisy_max_with_grumbel_noises(log_terms):
    """ Sampling an item via exponential mechanism

    Args:
      log_terms: Array of terms of form log(coefficient) - (exponent term).

    Returns:
      A sample from the exponential mechanism determined by terms.

    """
    noisy_log_scores = np.random.gumbel(scale=1, size=log_terms.shape) + log_terms
    winner = np.unravel_index(np.argmax(noisy_log_scores), noisy_log_scores.shape)
    min_time = noisy_log_scores[winner]
    if np.isnan(min_time) or np.isinf(min_time):
        raise RuntimeError(
            "Racing sample encountered inf or nan min time: {}".format(min_time))
    return winner


def sample_error_idx(log_error_counts_matrix, tau, epsilon, neighbor_type):
    """Samples a pair of (error, col) index by the exponential mechanism.

    Args:
      log_error_counts_matrix: a matrix log_sequence_count_matrix where, log_sequence_count_matrix[i][j] =
        log(# of sequences whose maximum error (truncated at the value of tau) equals i,
        and the first coordinate the maximum error occurs is given by j).
      tau: Pruning threshold, as defined in our paper.
      epsilon: Privacy parameter epsilon.
      neighbor_type: Available neighbor types are defined in the DP_Parameters

    Returns:
      (error, col) sampled with distribution
      P[(error, col)] ~ count[(error, col)] * exp(-epsilon * error / (2 * sensitivity)).
    """
    if neighbor_type is NeighborType.SWAP:
        sensitivity = 2
    else:
        sensitivity = 1

    range_array = epsilon * np.arange(tau + 1) / (2 * sensitivity)
    # Subtract the range array from each column of the matrix
    log_terms = log_error_counts_matrix - range_array[:, np.newaxis]
    winner = report_noisy_max_with_grumbel_noises(log_terms)
    return winner


def sample_swap_to_the_back(arr):
    # Randomly select an index
    random_index = np.random.randint(0, len(arr))

    # Swap the element at random_index with the last element
    arr[random_index], arr[-1] = arr[-1], arr[random_index]


def expand_candidates(candidates, sorted_indices_groups, index, threshold):
    for index in range(index, len(sorted_indices_groups)):
        if sorted_indices_groups[index][0] < threshold:
            break
        candidates.extend(sorted_indices_groups[index][1])
    return index


def sample_sequence(sorted_indices_groups, d, k, true_top_k_scores, tau, error, error_col):
    """
    Args:
      error_col:
      error:
      sorted_indices_groups:  a sorted list of tuples, in decreasing order, according to the first coordinate.
        Each element in the list is a tuple (val, [idx0, idx1, idx2, ...]), containing a value (val),
        and a list of indices (idx0, idx1, idx2, ...).
      d: An integer indicating the total number of items.
      true_top_k_scores: An array of the k largest scores, in decreasing order.
      k: An integer indicating the number of desired items.
      tau: A pruning threshold, as defined in our paper.

    Returns:
      A sequence sampled uniformly at random from those whose maximum error equals (error),
        and the first coordinate the maximum error occurs is given by (error_col).
    """
    sequence = []
    candidates = []
    index = 0
    for j in range(error_col):
        threshold = true_top_k_scores[j] - error + 1
        index = expand_candidates(candidates, sorted_indices_groups, index, threshold)

        sample_swap_to_the_back(candidates)
        sequence.append(candidates[-1])
        # Pop the last element from the array
        candidates.pop()

    threshold = true_top_k_scores[error_col] - error + 1
    index = expand_candidates(candidates, sorted_indices_groups, index, threshold)

    if error == tau:
        # Compute the set difference and convert to NumPy array
        not_sampled = set(range(d)) - set(sequence)
        candidates = list(not_sampled - set(candidates))
        # candidates = list(set(range(d)) - set(candidates) - set(sequence))
        sample_swap_to_the_back(candidates)
        sequence.append(candidates[-1])
        not_sampled.remove(candidates[-1])
        candidates = list(not_sampled)
        for j in range(error_col + 1, k):
            sample_swap_to_the_back(candidates)
            sequence.append(candidates[-1])
            candidates.pop()
    else:
        threshold = true_top_k_scores[error_col] - error
        assert sorted_indices_groups[index][0] == threshold
        sample_swap_to_the_back(sorted_indices_groups[index][1])
        sequence.append(sorted_indices_groups[index][1][-1])
        candidates.extend(sorted_indices_groups[index][1][:-1])
        index += 1

        for j in range(error_col + 1, k):
            threshold = true_top_k_scores[j] - error
            index = expand_candidates(candidates, sorted_indices_groups, index, threshold)

            sample_swap_to_the_back(candidates)
            sequence.append(candidates[-1])
            # Pop the last element from the array
            candidates.pop()

    return sequence


def fast_joint_sampling_dp_top_k(item_counts,
                                 k,
                                 epsilon,
                                 neighbor_type,
                                 failure_probability):
    """
    Args:
        item_counts:  a histogram
        neighbor_type: neighboring dataset type
        failure_probability: failure probability for sampling a sequence
            whose error score no less than the computed truncation threshold
        k: An integer indicating the number of desired items.
        epsilon: privacy parameter

    Returns:
      Array of k item indices as estimated by the fast joint exponential mechanism.
    """
    # print(failure_probability)
    hist = item_counts
    # d: An integer indicating the total number of items.
    d = len(hist)
    # find the true top-k scores in the histogram
    true_top_k_scores = sort_k_largest(hist, k)
    # compute the error search range and pruning threshold
    tau = compute_error_truncation_threshold(epsilon=epsilon, d=d, k=k, failure_probability=failure_probability,
                                             neighbor_type=neighbor_type)
    # print(tau)
    lowest_score_to_be_considered = true_top_k_scores[-1] - tau

    # Create a histogram of the item_counts
    # start_time = timeit.default_timer()
    index_groups = construct_index_groups(hist)
    # end_time = timeit.default_timer()
    # print("Time for index_groups: " + str(end_time - start_time))

    # start_time = timeit.default_timer()
    sorted_indices_groups = sort_group_indices_by_score_above_threshold(index_groups, lowest_score_to_be_considered)
    # end_time = timeit.default_timer()
    # print("Time for sorted_indices_groups: " + str(end_time - start_time))

    # compute log err counts matrix
    # start_time = timeit.default_timer()
    log_sequence_count_matrix = compute_log_sequence_count_matrix(true_top_k_scores, sorted_indices_groups, tau, d, k)
    # end_time = timeit.default_timer()
    # print("Time for computing sequence count matrix: " + str(end_time - start_time))

    # Assert that np.nan does not appear in the array
    # assert not np.isnan(log_sequence_count_matrix).any(), "np.nan appears in the array"

    # start_time = timeit.default_timer()
    error, error_col = sample_error_idx(log_sequence_count_matrix, tau, epsilon, neighbor_type)
    # end_time = timeit.default_timer()
    # print("Time for sampling index: " + str(end_time - start_time))

    # start_time = timeit.default_timer()
    sequence = sample_sequence(sorted_indices_groups, d, k, true_top_k_scores, tau, error, error_col)
    # end_time = timeit.default_timer()
    # print("Time for sampling sequence: " + str(end_time - start_time))

    return sequence


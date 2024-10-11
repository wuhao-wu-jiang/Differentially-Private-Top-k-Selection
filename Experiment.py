# coding=utf-8
# -----------------------------------------------------------------------------
# Derivative Work: Copyright 2024 Hao Wu, Hanwen Zhang.
#
# This file is a derivative of the file "experiment.py" from the open-source
# library dp_topk, authored by The Google Research Authors (as listed below).
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
#
# -----------------------------------------------------------------------------
# Original Work (dp_topk):
#     This file is a derivative of the file "experiment.py" from the
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

"""Differentially private top-k experiment and plotting code."""

import enum
import functools
import pickle
import timeit

import matplotlib.pyplot as plt
import numpy as np
import pandas

import FastTopk
from dp_topk import baseline_mechanisms
from dp_topk import joint


class TopKEstimationMethod(enum.Enum):
    JOINT = 1
    PNF_JOINT = 2
    CDP_PEEL = 3
    PNF_PEEL = 4
    LAP = 5
    GAMMA = 6
    FastTopK = 7


_PARTIAL_METHODS = {
    TopKEstimationMethod.JOINT: joint.joint,
    TopKEstimationMethod.PNF_JOINT: joint.pnf_joint,
    TopKEstimationMethod.CDP_PEEL: baseline_mechanisms.cdp_peeling_mechanism,
    TopKEstimationMethod.PNF_PEEL: baseline_mechanisms.pnf_peeling_mechanism,
    TopKEstimationMethod.LAP: baseline_mechanisms.laplace_mechanism,
    TopKEstimationMethod.GAMMA: baseline_mechanisms.gamma_mechanism,
    TopKEstimationMethod.FastTopK: FastTopk.fast_joint_sampling_dp_top_k,
}

_PLOT_LABELS = {
    TopKEstimationMethod.JOINT: "Joint",
    TopKEstimationMethod.PNF_JOINT: "PNF Joint",
    TopKEstimationMethod.CDP_PEEL: "CDP Peel",
    TopKEstimationMethod.PNF_PEEL: "PNF Peel",
    TopKEstimationMethod.LAP: "Laplace",
    TopKEstimationMethod.GAMMA: "Gamma",
    TopKEstimationMethod.FastTopK: "FastJoint",
}

_PLOT_LINESTYLES = {
    TopKEstimationMethod.JOINT: ":",
    TopKEstimationMethod.PNF_JOINT: ":",
    TopKEstimationMethod.CDP_PEEL: ":",
    TopKEstimationMethod.PNF_PEEL: ":",
    TopKEstimationMethod.LAP: "-.",
    TopKEstimationMethod.GAMMA: "-.",
    TopKEstimationMethod.FastTopK: ":",
}

_PLOT_MARKERS = {
    TopKEstimationMethod.JOINT: "*",
    TopKEstimationMethod.PNF_JOINT: "^",
    TopKEstimationMethod.CDP_PEEL: "P",
    TopKEstimationMethod.PNF_PEEL: "o",
    TopKEstimationMethod.LAP: "8",
    TopKEstimationMethod.GAMMA: "d",
    TopKEstimationMethod.FastTopK: "p",
}

_PLOT_MARKER_SIZES = {
    TopKEstimationMethod.JOINT: 9,
    TopKEstimationMethod.PNF_JOINT: 9,
    TopKEstimationMethod.CDP_PEEL: 8,
    TopKEstimationMethod.PNF_PEEL: 5,
    TopKEstimationMethod.LAP: 8,
    TopKEstimationMethod.GAMMA: 8,
    TopKEstimationMethod.FastTopK: 9,
}

_PLOT_COLORS = {
    TopKEstimationMethod.PNF_JOINT: "sienna",
    TopKEstimationMethod.FastTopK: "darkgreen",
    TopKEstimationMethod.JOINT: "C1",
    TopKEstimationMethod.CDP_PEEL: "C0",
    TopKEstimationMethod.PNF_PEEL: "C4",
    TopKEstimationMethod.LAP: "purple",
    TopKEstimationMethod.GAMMA: "darkorange",
}

_PLOT_FILL_COLORS = {
    TopKEstimationMethod.JOINT: "C1",
    TopKEstimationMethod.PNF_JOINT: "chocolate",
    TopKEstimationMethod.CDP_PEEL: "C0",
    TopKEstimationMethod.PNF_PEEL: "C4",
    TopKEstimationMethod.LAP: "violet",
    TopKEstimationMethod.GAMMA: "lightpink",
    TopKEstimationMethod.FastTopK: "green",
}


class ExperimentType(enum.Enum):
    COMPARE_K = 1
    COMPARE_EPSILON = 2
    COMPARE_FAILURE_PROBABILITY = 3


def linf_error(true_top_k, est_top_k):
    """Computes l_inf distance between the true and estimated top k counts.

    Args:
      true_top_k: Nonincreasing sequence of counts of true top k items.
      est_top_k: Sequence of counts of estimated top k items.

    Returns:
      l_inf distance between true_top_k and sequence.
    """
    return np.linalg.norm(true_top_k - est_top_k, ord=np.inf)


def l1_error(true_top_k, est_top_k):
    """Computes l_1 distance between the true and estimated top k counts.

    Args:
      true_top_k: Nonincreasing sequence of counts of true top k items.
      est_top_k: Sequence of counts of estimated top k items.

    Returns:
      l_1 distance between true_top_k and sequence.
    """
    return np.linalg.norm(true_top_k - est_top_k, ord=1)


def k_relative_error(true_top_k, est_top_k):
    """Computes k-relative error between the true and estimated top k counts.

    Args:
      true_top_k: Nonincreasing sequence of counts of true top k items.
      est_top_k: Sequence of counts of estimated top k items.

    Returns:
      max_{i in [k]} (c_k - c'_i), where c_1, ..., c_k are the true top k counts
      and c'_1, ..., c'_k are the estimated top k counts.
    """
    return np.amax(true_top_k[-1] - est_top_k)


def linf_error_idx(true_top_k, est_top_k):
    """Computes the index with the maximum error between true and estimated top k.

    Args:
      true_top_k: Nonincreasing sequence of counts of true top k items.
      est_top_k: Sequence of counts of estimated top k items.

    Returns:
      Index i such that |c_i - c'_i| = ||c_{:k} - c'_{:k}||_infty.
    """
    return np.argmax(np.abs(true_top_k - est_top_k))


class ErrorMetric(enum.Enum):
    L_INF = 1
    L_1 = 2
    K_REL = 3
    L_INF_IDX = 4


_ERROR_FUNCS = {
    ErrorMetric.L_INF: linf_error,
    ErrorMetric.L_1: l1_error,
    ErrorMetric.K_REL: k_relative_error,
    ErrorMetric.L_INF_IDX: linf_error_idx
}

_ERROR_LABELS = {
    ErrorMetric.L_INF: "$\\ell_\\infty$ error",
    ErrorMetric.L_1: "$\\ell_1$ error",
    ErrorMetric.K_REL: "$k$-relative error",
    ErrorMetric.L_INF_IDX: "$\\ell_\\infty$ error index"
}


def compare(item_counts, methods, d, default_k, default_epsilon, default_failure_probability,
            variable_range, variable_label, delta, num_trials,
            neighbor_type, experiment_type):
    """Computes 25th, 50th, and 75th percentile errors and times for each method.

    Args:
      experiment_type: Available experiment types are defined in the ExperimentType enum.
      variable_range: The range of the parameter to be tested
      variable_label: The label of the parameter to be tested
      default_failure_probability: Default failure probability for the pruning threshold for FastTopK
      default_epsilon: Default privacy parameter epsilon.
      default_k: Default value of k, the number of top items to estimate.
      item_counts: Array of item counts.
      methods: Available top-k estimation methods are defined in the
        TopKEstimationMethod enum.
      d: The number of counts subsampled uniformly at random from item_counts, or
        -1 to sample all counts.
      delta: Overall privacy parameter delta (only used for CDP peeling
        mechanism).
      num_trials: Number of trials to run for each k in k_range.
      neighbor_type: Available neighbor types are defined in the NeighborType
        enum.

    Returns:
      Dictionary results where results["time (s)"] is a
      (# methods) x (# variables in variable_range) x 3 array storing 0.25, 0.5, and 0.75
      quantile times, and for error_label in the _ERROR_LABELS enum,
      results[error_label] is a (# methods) x (# variables in variable_range) x 3 array storing
      0.25, 0.5, and 0.75 quantile errors for the corresponding error metric.

    Raises:
      ValueError: Unrecognized method name: [method].
    """
    num_variables = len(variable_range)
    num_methods = len(methods)
    quantiles = [0.25, 0.5, 0.75]
    num_quantiles = 3
    errors = np.empty((len(ErrorMetric), num_methods, num_variables, num_quantiles))
    times = np.empty((num_methods, num_variables, num_quantiles))
    raw_errors = np.empty((len(ErrorMetric), num_methods, num_variables, num_trials))
    raw_times = np.empty((num_methods, num_variables, num_trials))
    if d == -1:
        d = len(item_counts)
    item_counts_generator = lambda: np.random.permutation(item_counts)[:d]
    method_fns = []
    for method in methods:
        if experiment_type == ExperimentType.COMPARE_K:
            method_fn = functools.partial(_PARTIAL_METHODS[method], epsilon=default_epsilon)
        elif experiment_type == ExperimentType.COMPARE_EPSILON:
            method_fn = functools.partial(_PARTIAL_METHODS[method], k=default_k)
        elif experiment_type == ExperimentType.COMPARE_FAILURE_PROBABILITY:
            method_fn = functools.partial(_PARTIAL_METHODS[method], epsilon=default_epsilon, k=default_k)
        else:
            raise ValueError("Unrecognized Experiment Type: {}".format(experiment_type))

        if method == TopKEstimationMethod.JOINT or method == TopKEstimationMethod.PNF_JOINT:
            method_fn = functools.partial(method_fn, neighbor_type=neighbor_type)
        elif method == TopKEstimationMethod.CDP_PEEL:
            method_fn = functools.partial(method_fn, delta=delta)
        elif method == TopKEstimationMethod.LAP:
            method_fn = functools.partial(method_fn, c=d, neighbor_type=neighbor_type)
        elif method == TopKEstimationMethod.FastTopK:
            method_fn = functools.partial(method_fn, neighbor_type=neighbor_type)
            if experiment_type != ExperimentType.COMPARE_FAILURE_PROBABILITY:
                method_fn = functools.partial(method_fn, failure_probability=default_failure_probability)
        elif method != TopKEstimationMethod.PNF_PEEL and method != TopKEstimationMethod.GAMMA:
            raise ValueError("Unrecognized method name: {}".format(method))
        method_fns.append(method_fn)
    for variable_idx in range(num_variables):
        variable = variable_range[variable_idx]
        print("running " + variable_label + ": " + str(variable))
        variable_errors = np.empty((len(ErrorMetric), num_methods, num_trials))
        variable_times = np.empty((num_methods, num_trials))
        for trial in range(num_trials):
            item_counts = item_counts_generator()
            if experiment_type == ExperimentType.COMPARE_K:
                true_top_k = np.sort(item_counts)[::-1][:variable]
            else:
                true_top_k = np.sort(item_counts)[::-1][:default_k]
            for method_idx in range(num_methods):
                if experiment_type == ExperimentType.COMPARE_K:
                    start_time = timeit.default_timer()
                    selected_items = method_fns[method_idx](item_counts=item_counts, k=variable)
                    end_time = timeit.default_timer()
                elif experiment_type == ExperimentType.COMPARE_EPSILON:
                    start_time = timeit.default_timer()
                    selected_items = method_fns[method_idx](item_counts=item_counts, epsilon=variable)
                    end_time = timeit.default_timer()
                elif experiment_type == ExperimentType.COMPARE_FAILURE_PROBABILITY:
                    if methods[method_idx] == TopKEstimationMethod.FastTopK:
                        start_time = timeit.default_timer()
                        selected_items = method_fns[method_idx](item_counts=item_counts, failure_probability=variable)
                        end_time = timeit.default_timer()
                    else:
                        start_time = timeit.default_timer()
                        selected_items = method_fns[method_idx](item_counts=item_counts)
                        end_time = timeit.default_timer()
                else:
                    raise ValueError("Unrecognized Experiment Type: {}".format(experiment_type))

                # variable_times[method_idx][trial] = end - start
                variable_times[method_idx][trial] = end_time - start_time
                for metric in ErrorMetric:
                    variable_errors[metric.value - 1][method_idx][trial] = _ERROR_FUNCS[metric](
                        true_top_k, item_counts[selected_items])
        for method_idx in range(num_methods):
            times[method_idx][variable_idx] = np.quantile(variable_times[method_idx], quantiles)
            raw_times[method_idx][variable_idx] = variable_times[method_idx]
            for metric in ErrorMetric:
                errors[metric.value - 1][method_idx][variable_idx] = np.quantile(
                    variable_errors[metric.value - 1][method_idx], quantiles)
                raw_errors[metric.value - 1][method_idx][variable_idx] = variable_errors[metric.value - 1][method_idx]
    results = {}
    raw_results = {}
    results["time (s)"] = times
    raw_results["time (s)"] = raw_times
    for metric in ErrorMetric:
        results[_ERROR_LABELS[metric]] = errors[metric.value - 1]
        raw_results[_ERROR_LABELS[metric]] = raw_errors[metric.value - 1]
    return results, raw_results


def save_meta(counts, data_source):
    df = pandas.DataFrame([[len(counts), np.amin(counts), np.amax(counts)]],
                          columns=["#" + data_source, "min score", "max score"])
    # File path to save the CSV
    file_path = "metadata/" + "meta_" + data_source + ".csv"
    # Save the DataFrame as a CSV file
    df.to_csv(file_path, index=False)


def save_python_variables(k_results, raw_k_results, eps_results, raw_eps_results, failure_probability_results,
                          raw_failure_probability_results, data_source):
    filename = "raw_results/" + data_source + ".pkl"
    # Save the dictionary to a file
    with open(filename, 'wb') as f:
        pickle.dump([k_results, raw_k_results, eps_results, raw_eps_results, failure_probability_results,
                     raw_failure_probability_results], f)


def load_python_variables(data_source):
    filename = "raw_results/" + data_source + ".pkl"
    # load variables from filename
    with open(filename, 'rb') as f:
        k_results, raw_k_results, \
            eps_results, raw_eps_results, \
            failure_probability_results, raw_failure_probability_results = pickle.load(f)
    return k_results, raw_k_results, \
        eps_results, raw_eps_results, failure_probability_results, raw_failure_probability_results


def test_load_python_variables(k_results, raw_k_results, eps_results, raw_eps_results, failure_probability_results,
                               raw_failure_probability_results, data_source, methods):
    load_k_results, load_raw_k_results, \
        load_eps_results, load_load_raw_eps_results, \
        load_load_failure_probability_results, load_raw_failure_probability_results = load_python_variables(data_source)
    for metric in ErrorMetric:
        np.testing.assert_array_equal(k_results[_ERROR_LABELS[metric]],
                                      load_k_results[_ERROR_LABELS[metric]])
        np.testing.assert_array_equal(raw_k_results[_ERROR_LABELS[metric]],
                                      load_raw_k_results[_ERROR_LABELS[metric]])
        np.testing.assert_array_equal(eps_results[_ERROR_LABELS[metric]],
                                      load_eps_results[_ERROR_LABELS[metric]])
        np.testing.assert_array_equal(raw_eps_results[_ERROR_LABELS[metric]],
                                      load_load_raw_eps_results[_ERROR_LABELS[metric]])
        np.testing.assert_array_equal(failure_probability_results[_ERROR_LABELS[metric]],
                                      load_load_failure_probability_results[_ERROR_LABELS[metric]])
        np.testing.assert_array_equal(raw_failure_probability_results[_ERROR_LABELS[metric]],
                                      load_raw_failure_probability_results[_ERROR_LABELS[metric]])

    print("Pass Test For Variable Saving")


def run_experiment(meta_compare_fn, meta_plot_fn, counts, data_source, k_range, eps_range, failure_probability_range,
                   methods):
    compare_fn = functools.partial(meta_compare_fn, item_counts=counts, d=-1)
    plot_fn = functools.partial(meta_plot_fn, data_source=data_source, output_folder="plots")
    save_parameter_range_fn = functools.partial(save_parameter_range, data_source=data_source,
                                                output_folder="results", methods=methods)

    k_results, raw_k_results = compare_fn(variable_range=k_range, variable_label="k",
                                          experiment_type=ExperimentType.COMPARE_K)
    plot_fn(results=k_results, parameter_range=k_range, parameter_label="k", log_y_axis=True)
    save_parameter_range_fn(results=k_results, parameter_range=k_range, parameter_label="k")

    eps_results, raw_eps_results = compare_fn(variable_range=eps_range, variable_label="eps",
                                              experiment_type=ExperimentType.COMPARE_EPSILON)
    plot_fn(results=eps_results, parameter_range=eps_range, parameter_label="eps", log_y_axis=True)
    save_parameter_range_fn(results=eps_results, parameter_range=eps_range, parameter_label="eps")

    failure_probability_results, raw_failure_probability_results \
        = compare_fn(variable_range=failure_probability_range, variable_label="beta",
                     experiment_type=ExperimentType.COMPARE_FAILURE_PROBABILITY)

    plot_fn(results=failure_probability_results, parameter_range=failure_probability_range, parameter_label="beta",
            log_y_axis=False, log_x_axis=True)
    save_parameter_range_fn(results=failure_probability_results, parameter_range=failure_probability_range,
                            parameter_label="beta")

    save_python_variables(k_results, raw_k_results, eps_results, raw_eps_results, failure_probability_results,
                          raw_failure_probability_results, data_source)

    # test_load_python_variables(k_results, raw_k_results, eps_results, raw_eps_results, failure_probability_results,
    #                            raw_failure_probability_results, data_source, methods)


def plot_parameter_range(data_source, output_folder, methods, results, parameter_range, log_y_axis, legend,
                         parameter_label, log_x_axis=False):
    """Plots errors and times data generated by compare and saves plots as .png.

    Args:
      output_folder: Output folder for the figures
      parameter_label: Label for the variable
      log_x_axis: Boolean determining whether plot x-axis is logarithmic.
      parameter_range: Range for the variable
      data_source: Data source used to generate input results.
      methods: Top-k estimation methods used to generate input results. Available
        top-k estimation methods are defined in the TopKEstimationMethod enum.
      results: Dictionary of error and time data generated by compare.
      log_y_axis: Boolean determining whether plot y-axis is logarithmic.
      legend: Boolean determining whether the legend appears.

    Returns:
      An error plot for each error metric and one time plot. Each error plot is
      saved as $data_source_error_metric.png where error_metric is defined in
      ErrorMetric.name, and the time plot is saved as $data_source_time.png.
    """
    line_opacity = 0.9
    confidence_interval_opacity = 0.25
    marker_size = 9
    line_width = 2
    # a parameter to avoid log (0) in log plot
    y_log_plot_shift = 10 * (10 ** (-1)) if log_y_axis else 0

    for metric in ErrorMetric:
        # plt.xlabel("k", fontsize=20)
        if parameter_label == "eps":
            plt.xlabel("$\\epsilon$", fontsize=18)
        elif parameter_label == "beta":
            plt.xlabel("$\\beta$", fontsize=18)
        else:
            plt.xlabel("$k$", fontsize=18)

        ax = plt.gca()
        ax.tick_params(labelsize=18)
        if log_x_axis:
            plt.xscale("log")
        if log_y_axis:
            plt.yscale("log")
        # plt.title(data_source + " " + _ERROR_LABELS[metric], fontsize=20)
        plt.title(data_source, fontsize=20)
        for method_idx in range(len(methods)):
            method = methods[method_idx]
            plt.plot(
                parameter_range,
                # results[_ERROR_LABELS[metric]][method_idx, :, 1] + 1,
                results[_ERROR_LABELS[metric]][method_idx, :, 1] + y_log_plot_shift,
                linestyle=_PLOT_LINESTYLES[method],
                marker=_PLOT_MARKERS[method],
                label=_PLOT_LABELS[method],
                color=_PLOT_COLORS[method],
                linewidth=line_width,
                alpha=line_opacity,
                markerfacecolor='none',
                ms=_PLOT_MARKER_SIZES[method]
            )
            plt.fill_between(
                parameter_range,
                # results[_ERROR_LABELS[metric]][method_idx, :, 0] + 1,
                results[_ERROR_LABELS[metric]][method_idx, :, 0] + y_log_plot_shift,
                # results[_ERROR_LABELS[metric]][method_idx, :, 2] + 1,
                results[_ERROR_LABELS[metric]][method_idx, :, 2] + y_log_plot_shift,
                color=_PLOT_FILL_COLORS[method],
                alpha=confidence_interval_opacity)
        if legend:
            ax.legend(
                loc="lower center",
                bbox_to_anchor=(0.45, -0.4),
                ncol=3,
                frameon=False,
                fontsize=16)
        plt.ylabel(_ERROR_LABELS[metric], fontsize=20)
        plt.savefig(
            output_folder + "/" + data_source + "_" + "var_" + parameter_label + "_" + str(metric.name) + ".pdf",
            bbox_inches="tight")
        plt.close()
    # plt.xlabel("k", fontsize=20)
    # plt.xlabel(parameter_label, fontsize=20)
    if parameter_label == "eps":
        plt.xlabel("$\\epsilon$", fontsize=18)
    elif parameter_label == "beta":
        plt.xlabel("$\\beta$", fontsize=18)
    else:
        plt.xlabel("$k$", fontsize=18)
    ax = plt.gca()
    ax.tick_params(labelsize=18)
    if log_x_axis:
        plt.xscale("log")
    # if log_y_axis:
    plt.yscale("log")
    plt.title(data_source + " " + _ERROR_LABELS[metric], fontsize=20)
    for method_idx in range(len(methods)):
        method = methods[method_idx]
        # print(results["time (s)"][method_idx, :, 1])
        plt.plot(
            parameter_range,
            results["time (s)"][method_idx, :, 1],
            linestyle=_PLOT_LINESTYLES[method],
            marker=_PLOT_MARKERS[method],
            label=_PLOT_LABELS[method],
            color=_PLOT_COLORS[method],
            linewidth=line_width,
            alpha=line_opacity,
            markerfacecolor='none',
            ms=_PLOT_MARKER_SIZES[method]
        )
        plt.fill_between(
            parameter_range,
            results["time (s)"][method_idx, :, 0],
            results["time (s)"][method_idx, :, 2],
            color=_PLOT_FILL_COLORS[method],
            alpha=confidence_interval_opacity)
        plt.ylabel(_ERROR_LABELS[metric], fontsize=20)
        # plt.title(data_source + " time", fontsize=20)
        plt.title(data_source, fontsize=20)
    if legend:
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.45, -0.4),
            ncol=3,
            frameon=False,
            fontsize=16)
    plt.ylabel("time (s)", fontsize=20)
    plt.savefig(output_folder + "/" + data_source + "_" + "var_" + parameter_label + "_time.pdf", bbox_inches="tight")
    plt.close()

    if not legend:
        fig_size = (6, 1)
        fig_leg = plt.figure(figsize=fig_size)
        ax_leg = fig_leg.add_subplot(111)
        # add the legend from the previous axes
        ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', ncol=7, frameon=False,
                      fontsize=16)
        # hide the axes frame and the x/y labels
        ax_leg.axis('off')
        # plt.tight_layout()
        fig_leg.savefig(output_folder + "/" + 'legend.pdf', bbox_inches="tight")
        plt.close()


def save_parameter_range(data_source, output_folder, methods, results, parameter_range, parameter_label):
    """Save errors and times data generated by compare and saves plots as .csv

    Args:
      output_folder: Output folder for the figures
      parameter_label: Label for the variable
      parameter_range: Range for the variable
      data_source: Data source used to generate input results.
      methods: Top-k estimation methods used to generate input results. Available
        top-k estimation methods are defined in the TopKEstimationMethod enum.
      results: Dictionary of error and time data generated by compare.

    Returns:
      An error plot for each error metric and one time plot. Each error plot is
      saved as $data_source_error_metric.png where error_metric is defined in
      ErrorMetric.name, and the time plot is saved as $data_source_time.png.
    """
    quantiles = [0.25, 0.5, 0.75]
    row_names = [_PLOT_LABELS[method] for method in methods]
    for metric in ErrorMetric:
        for i, quantile in enumerate(quantiles):
            # Convert data to a DataFrame
            df = pandas.DataFrame(results[_ERROR_LABELS[metric]][:, :, i], columns=parameter_range)
            # Set row names for the DataFrame
            df.index = row_names
            # File path to save the CSV
            file_path = output_folder + "/" + data_source + "_" + "var_" + parameter_label + "_" + str(
                metric.name) + "_" + str(quantile) + ".csv"
            # Save the DataFrame as a CSV file
            df.to_csv(file_path, index=True)

    for i, quantile in enumerate(quantiles):
        # Convert data to a DataFrame
        df = pandas.DataFrame(results["time (s)"][:, :, i], columns=parameter_range)
        # Set row names for the DataFrame
        df.index = row_names
        # File path to save the CSV
        file_path = output_folder + "/" + data_source + "_" + "var_" + parameter_label + "_" + str(
            quantile) + "_time.csv"
        # Save the DataFrame as a CSV file
        df.to_csv(file_path, index=True)


def counts_histogram(item_counts, plot_title, plot_name):
    """Computes and plots histogram of item counts.

    Args:
      item_counts: Array of item counts.
      plot_title: Plot title.
      plot_name: Plot will be saved as plot_name.png.

    Returns:
      Histogram of item counts using 100 bins.
    """
    plt.title(plot_title, fontsize=20)
    plt.xlabel("item count", fontsize=20)
    plt.ylabel("# items", fontsize=20)
    plt.yscale("log")
    ax = plt.gca()
    ax.tick_params(labelsize=18)
    plt.hist(item_counts, bins=100)
    plt.savefig(plot_name + ".png")
    plt.close()


def compute_and_plot_diffs(item_counts, d, k_range, num_trials, log_y_axis,
                           plot_title, plot_name):
    """Computes and plots median diffs between top k counts.

    Args:
      item_counts: Array of item counts.
      d: Total number of items to subsample from data in each trial.
      k_range: Range for k, the number of top items estimated.
      num_trials: Number of trials to average over.
      log_y_axis: Boolean determining whether plot y-axis is logarithmic.
      plot_title: Title displayed on the plot.
      plot_name: Plot will be saved as plot_name.png.

    Returns:
      Plot of median diff between k^{th} and (k+1}^{th} sorted item count
      for each k in k_range, where each trial subsamples min(data size, d) counts.
    """
    diffs = np.zeros((num_trials, len(k_range)))
    for trial in range(num_trials):
        sample = np.sort(np.random.permutation(item_counts)[:d])[::-1]
        trial_diffs = sample[:-1] - sample[1:]
        diffs[trial] = trial_diffs[k_range]
    median_diffs = np.quantile(diffs, q=0.5, axis=0)
    if log_y_axis:
        plt.yscale("log")
    plt.xlabel("$k$", fontsize=20)
    plt.ylabel("count diff", fontsize=20)
    ax = plt.gca()
    ax.tick_params(labelsize=18)
    plt.title(plot_title, fontsize=20)
    plt.plot(k_range, 1 + median_diffs,
             linewidth=3,
             linestyle=":",
             marker=".",
             color="C1",
             markerfacecolor='none',
             ms=1)
    plt.savefig(plot_name + ".pdf", bbox_inches="tight")
    plt.close()

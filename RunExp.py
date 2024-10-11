# coding=utf-8
# -----------------------------------------------------------------------------
# Derivative Work: Copyright 2024 Hao Wu, Hanwen Zhang.
#
# This file is a derivative of the file "run_experiment.py" from the open-source
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
#     This file is a derivative of the file "run_experiment.py" from the
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



"""Script for running single experiment and saving its plots."""
import functools
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import DP_Parameters
import Experiment

methods = [
    Experiment.TopKEstimationMethod.FastTopK,
    Experiment.TopKEstimationMethod.JOINT,
    Experiment.TopKEstimationMethod.CDP_PEEL,
    Experiment.TopKEstimationMethod.PNF_PEEL,
]

# k_range = np.arange(90, 110, 10)
k_range = np.arange(10, 210, 10)
eps_range = np.array([1.0 / 4, 1.0 / 2, 1, 2, 4])
failure_probability_range = 2.0 ** (np.arange(start=-6, stop=-16, step=-2))
default_k = 100
# default_k_idx = np.where(k_range == default_k)[0][0]
default_eps = 1.0
default_failure_probability = 2.0 ** (-10)
delta = 1e-6
# num_trials = 10
num_trials = 200

meta_compare_fn = functools.partial(Experiment.compare, methods=methods, default_k=default_k,
                                    default_epsilon=default_eps,
                                    default_failure_probability=default_failure_probability,
                                    delta=delta, num_trials=num_trials,
                                    neighbor_type=DP_Parameters.NeighborType.ADD_REMOVE)
meta_plot_fn = functools.partial(Experiment.plot_parameter_range, output_folder="plots", methods=methods, legend=False)

run_experiment_fn = functools.partial(Experiment.run_experiment, meta_compare_fn=meta_compare_fn,
                                      meta_plot_fn=meta_plot_fn, k_range=k_range, eps_range=eps_range,
                                      failure_probability_range=failure_probability_range, methods=methods)

dataset_ids_to_run = set([int(x) for x in sys.argv[1:]])

# Get the current date and time
current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d-%HH-%MM")
# Open the file in write mode
with open("running_info/" + str(formatted_datetime) + ".txt", 'w') as f:
    f.write("System Input: " + str(sys.argv[1:]) + "\n")
    f.write("k_range: " + str(list(k_range)) + "\n")
    f.write("eps_range: " + str(list(eps_range)) + "\n")
    f.write("failure_probability_range: " + str(list(failure_probability_range)) + "\n")
    f.write("default_k: " + str(default_k) + "\n")
    # f.write("default_k_idx: " + str(default_k_idx) + "\n")
    f.write("default_eps: " + str(default_eps) + "\n")
    f.write("default_failure_probability: " + str(default_failure_probability) + "\n")
    f.write("delta: " + str(delta) + "\n")
    f.write("num_trials: " + str(num_trials) + "\n")
    f.write("methods: " + str(methods) + "\n")

if 0 in dataset_ids_to_run:
    books = pd.read_csv("datasets/books.csv", usecols=["ratings_count"])
    counts = np.array(books["ratings_count"][1:]).astype(int)
    data_source = "books"
    Experiment.save_meta(counts, data_source)
    run_experiment_fn(data_source=data_source, counts=counts)

# Instructions for the other five datasets in the paper appear below.

# games: https://www.kaggle.com/tamber/steam-video-games/version/3
# Save the file as games.csv and use
if 1 in dataset_ids_to_run:
    column_names = ['user', 'game', 'behavior', 'hours_or_bool', '?']
    games = pd.read_csv("datasets/games.csv", names=column_names, skipinitialspace=False)
    counts = np.asarray(
        games.loc[games['behavior'] == 'purchase']['game'].value_counts())
    data_source = "games"
    Experiment.save_meta(counts, data_source)
    run_experiment_fn(data_source=data_source, counts=counts)

# news: https://archive.ics.uci.edu/ml/datasets/online+news+popularity
# Save the file as news.csv and use
if 2 in dataset_ids_to_run:
    news = pd.read_csv("datasets/news.csv", usecols=[" shares"])
    counts = np.array(news[" shares"]).astype(int)
    data_source = "news"
    Experiment.save_meta(counts, data_source)
    run_experiment_fn(data_source=data_source, counts=counts)

# movies: https://grouplens.org/datasets/movielens/25m/
# Save the file as movies.csv and use
if 3 in dataset_ids_to_run:
    movies = pd.read_csv("datasets/movies.csv", usecols=["movieId"])
    counts = movies.value_counts()
    data_source = "movies"
    Experiment.save_meta(counts, data_source)
    run_experiment_fn(data_source=data_source, counts=counts)

# tweets: https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/JBXKFD/F4FULO&version=2.2
if 4 in dataset_ids_to_run:
    tweets = pd.read_csv("datasets/tweets.csv", usecols=["number_of_likes"])
    counts = np.array(tweets["number_of_likes"]).astype(int)
    data_source = "tweets"
    Experiment.save_meta(counts, data_source)
    run_experiment_fn(data_source=data_source, counts=counts)

# foods: https://jmcauley.ucsd.edu/data/amazon/
# Save the file as foods.csv and use
if 5 in dataset_ids_to_run:
    column_names = ['user', 'item', 'rating', 'timestamp']
    food = pd.read_csv("datasets/foods.csv", names=column_names, skipinitialspace=False)
    counts = np.asarray(food['item'].value_counts())
    data_source = "food"
    Experiment.save_meta(counts, data_source)
    run_experiment_fn(data_source=data_source, counts=counts)

if 6 in dataset_ids_to_run:
    k_range = np.arange(10, 210, 10)

    books = pd.read_csv("datasets/books.csv", usecols=["ratings_count"])
    counts = np.array(books["ratings_count"][1:]).astype(int)
    data_source = "books"
    Experiment.save_meta(counts, data_source)
    Experiment.compute_and_plot_diffs(counts, len(counts), k_range, num_trials, log_y_axis=True,
                           plot_title=data_source, plot_name="plots/" + data_source + "_data_dist")

    # games: https://www.kaggle.com/tamber/steam-video-games/version/3

    column_names = ['user', 'game', 'behavior', 'hours_or_bool', '?']
    games = pd.read_csv("datasets/games.csv", names=column_names, skipinitialspace=False)
    counts = np.asarray(
        games.loc[games['behavior'] == 'purchase']['game'].value_counts())
    data_source = "games"
    Experiment.save_meta(counts, data_source)
    Experiment.compute_and_plot_diffs(counts, len(counts), k_range, num_trials, log_y_axis=True,
                                      plot_title=data_source, plot_name="plots/" + data_source + "_data_dist")

    # news: https://archive.ics.uci.edu/ml/datasets/online+news+popularity

    news = pd.read_csv("datasets/news.csv", usecols=[" shares"])
    counts = np.array(news[" shares"]).astype(int)
    data_source = "news"
    Experiment.save_meta(counts, data_source)
    Experiment.compute_and_plot_diffs(counts, len(counts), k_range, num_trials, log_y_axis=True,
                                      plot_title=data_source, plot_name="plots/" + data_source + "_data_dist")

    # movies: https://grouplens.org/datasets/movielens/25m/

    movies = pd.read_csv("datasets/movies.csv", usecols=["movieId"])
    counts = movies.value_counts()
    data_source = "movies"
    Experiment.save_meta(counts, data_source)
    Experiment.compute_and_plot_diffs(counts, len(counts), k_range, num_trials, log_y_axis=True,
                                      plot_title=data_source, plot_name="plots/" + data_source + "_data_dist")

    # tweets: https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/JBXKFD/F4FULO&version=2.2

    tweets = pd.read_csv("datasets/tweets.csv", usecols=["number_of_likes"])
    counts = np.array(tweets["number_of_likes"]).astype(int)
    data_source = "tweets"
    Experiment.save_meta(counts, data_source)
    Experiment.compute_and_plot_diffs(counts, len(counts), k_range, num_trials, log_y_axis=True,
                                      plot_title=data_source, plot_name="plots/" + data_source + "_data_dist")

    # foods: https://jmcauley.ucsd.edu/data/amazon/
    column_names = ['user', 'item', 'rating', 'timestamp']
    food = pd.read_csv("datasets/foods.csv", names=column_names, skipinitialspace=False)
    counts = np.asarray(food['item'].value_counts())
    data_source = "food"
    Experiment.save_meta(counts, data_source)
    Experiment.compute_and_plot_diffs(counts, len(counts), k_range, num_trials, log_y_axis=True,
                                      plot_title=data_source, plot_name="plots/" + data_source + "_data_dist")


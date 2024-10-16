# Differentially Private Top-k Selection

This repository provides an implementation of the **FastJoint** algorithm for differentially private top-k selection, as detailed in the paper:

> **Faster Differentially Private Top-k Selection: A Joint Exponential Mechanism with Pruning**

## Interface

The implementation can be found in `FastTopk.py` under the following function:

```python
def fast_joint_sampling_dp_top_k(item_counts, k, epsilon, neighbor_type, failure_probability)
```

**Arguments:**

- **item_counts**: A 1D numpy array representing the histogram (non-negative integer counts or scores for the items).
- **k**: The number of items to select.
- **epsilon**: The privacy parameter.
- **neighbor_type**: The type of neighboring dataset. Currently supports `"DP_Parameters.NeighborType.ADD_REMOVE"` as defined in `DP_Parameters.py`.
- **failure_probability**: The probability that the algorithm will return a sequence whose error exceeds the truncation threshold. The default value used in the experiments is $2^{-10}$.

**Returns:**  
An array containing the indices of the top `k` items selected by the FastJoint algorithm.

#### Example Usage
```python
import numpy as np

import DP_Parameters
from FastTopk import fast_joint_sampling_dp_top_k

# Create a histogram with counts from 20 to 2000 (1 to 100 multiplied by 20)
hist = np.arange(1, 101) * 20
k = 10
epsilon = 1
failure_probability = 2 ** (-10)

# Get the top k items using the FastJoint algorithm
top_k_items = fast_joint_sampling_dp_top_k(item_counts=hist, k=k, epsilon=epsilon,
                                           neighbor_type=DP_Parameters.NeighborType.ADD_REMOVE,
                                           failure_probability=failure_probability)

print(top_k_items)
```


## Background

This section provides a brief overview of the problem and the algorithms used. For detailed information, please refer to the paper.

Given $d$ items indexed by $[d] = \lbrace 1, \ldots, d \rbrace$ and a histogram $\vec{h} = (\vec{h}[1], \ldots, \vec{h}[d]) \in \mathbb{N}_+^d$ representing their counts or scores, the algorithm returns an ordered sequence $\vec{s} = (\vec{s}[1], \ldots, \vec{s}[k])$ of $k$ distinct items from $[d]$, approximating the largest scores while ensuring differential privacy.

#### Neighboring Relation

Differential privacy relies on defining neighboring datasets. Two histograms $\vec{h}, \vec{h}' \in \mathbb{N}_+^d$ are considered neighbors if they differ by at most $1$ in at most one entry.


#### Algorithm

The **FastJoint** algorithm samples a sequence $\vec{s} = (\vec{s}[1], \ldots, \vec{s}[k])$ directly from the collection of $d^{\Theta(k)}$ possible length-$k$ sequences, using the *exponential mechanism*, based on the following loss function:

$$
    \mathcal{E}rr(\vec{h}, \vec{s}) 
    \doteq \max_{i \in [k]} \left( \vec{h}_{(i)} - \vec{h} \left[ \vec{s}[i] \right] \right),
$$

where $\vec{h}_{(i)}$ is the $i^{th}$ largest entry in $\vec{h}$. The algorithm runs in $O\left( d + \frac{k^2}{\varepsilon} \cdot \ln d \right)$ time, where $\varepsilon$ is the privacy parameter.





## Empirical Evaluation

This section provides instructions for reproducing the empirical results presented in the paper.
It compares the **FastJoint** algorithm with the **Joint**, **CDP Peel**, and **PNF Peel** algorithms.
The implementations of the later three algorithms can be found in the public [dp_topk repository](https://github.com/google-research/google-research/tree/master/dp_topk) (as of May 2024).
Please follow the instructions below to set up the environment, and download the necessary datasets, to replicate the results.


### Code Integration

To integrate our modifications with the original `dp_topk` repository:

1. Clone the [dp_topk repository](https://github.com/google-research/google-research/tree/master/dp_topk).
2. Copy the files `baseline_mechanisms.py`, `differential_privacy.py` and `joint.py` from this repository.
3. Place them into the `dp_topk` folder.


### Datasets

To replicate our results, first download the datasets from the following sources:

1. **Goodreads Books**: [Goodreads-books](https://www.kaggle.com/jealousleopard/goodreadsbooks)
2. **Steam Video Games**: [Steam Video Games](https://www.kaggle.com/datasets/tamber/steam-video-games/data)
3. **Tweets Dataset**: [Tweets Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JBXKFD)  
4. **Online News Popularity**: [UCI Online News Popularity](https://archive.ics.uci.edu/dataset/332/online+news+popularity)  
5. **MovieLens 25M Dataset**: [MovieLens 25M Dataset](https://grouplens.org/datasets/movielens/25m/)
6. **Amazon Product Data (2014)**: [Amazon Grocery and Gourmet Food](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)

**Save the datasets** as:

- `books.csv` for the Goodreads dataset
- `games.csv` for the Steam Video Games dataset
- `tweets.csv` for the Tweets dataset
- `news.csv` for the Online News Popularity dataset
- `movies.csv` for the MovieLens dataset 
- `foods.csv` for the Amazon dataset

Place each file into a folder named `datasets`.

**Note:**  
For the MovieLens 25M Dataset, the download will provide a zip file named `ml-25m.zip`. Extract its contents and locate the file `ratings.csv`. Rename this file to `movies.csv` and move it to the `datasets` folder.


### Running the Experiments

To run the experiments, execute the following command:

```bash
python3 RunExp.py [num]
```

where `[num]` is an integer corresponding to a specific dataset:

| Value | Dataset     |
|-------|-------------|
| 0     | books       |
| 1     | games       |
| 2     | news        |
| 3     | movies      |
| 4     | tweets      |
| 5     | food        |



The current code repeats each experiment 200 times. To speed up the code for obtaining preliminary results, you can reduce the number of trials by changing the initialization `num_trials = 200` in `RunExp.py` to a smaller value.

For further details, refer to the file `RunExp.py` in the root directory.



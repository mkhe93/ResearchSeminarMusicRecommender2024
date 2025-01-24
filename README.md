# Research Seminar 2024
 __Note:__ this work and theory have been extended in the work / repository: [Application of Graph Neural Networks
to Music Recommender Systems](https://github.com/mkhe93/Thesis-GNN-Rec-2025)

## Project Structure

- [Introduction](#️-introduction)
- [Dataset](#-dataset)
- [Evaluation](#-evaluation)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
  - [Download the Data](#download-and-store-the-dataset)
  - [Setup the Virtual Environment](#setup-virtual-environment)
    - [Setup with conda](#setup-with-conda-recommended)
    - [Setup with pip](#setup-with-pip)
- [Export requirements.txt](#export-requirementstxt)
  - [For Conda](#for-conda)
  - [For Pip](#for-pip)

#

## 🕵️ Introduction

This repository contains code and resources for the research seminar titled "Non-Deep Learning Collaborative Filtering Techniques for Implicit Feedback Data in Music Recommender Systems." The focus of this seminar is on applying user-based and item-based neighborhood techniques, as well as matrix factorization using alternating least squares (ALS) optimization, to a real-life dataset with binary feedback. By implementing and evaluating these classical collaborative filtering methods, this work aims to establish a baseline for future research in music recommender systems, facilitating the development and comparison of more advanced techniques.

## 💾 Dataset

- \# of customers: 58.747
- \# of records: 37.370


| \#    | file name | \# rows, (users, items) | sparsity | features |
| ----- | --------- | ----------------------- | -------- | ---------|
| **1** | `user_item_interaction_RAW_ANONYMIZED.txt` | 23.545.542, (70.309, 37.408) | 0.68 % | userID, itemID   |
| **2** | `user_item_interaction_FILTERED_ANONYMIZED.txt` | 17.665.904, ( 58.747, 37.370 ) | 0.8 % | userID, itemID |

## 💎 Evaluation

After the hyperparameter search, the best 3 settings for each algorithm were applied to the entire filtered dataset. A Top-10 recommendation was made for a random set of 100 users. This process was repeated 5 times for each algorithm and each parameter setting, and the mean of the best 5 runs is reported in the following table in descending order by the NDCG. The best values are printed in bold and the second largest are underlined:


| Algorithm | Pre | MRR | nDCG | MAP | IC | ARP | APLT | 
| --------- | --- | --- | ---- | --- | -- | --- | ---- | 
|ALS-MF | __0.116__ | __0.3587__ | __0.4040__ | __0.2643__ |  __0.01663__ | 5179 |  0.00 |
|UserkNN |  <u>0.111</u> |  <u>0.3288</u> |  <u>0.3934</u> |  0.2544 |  0.01540 | 5831 |   0.002 |
|UserAsymkNN |  0.0980 | 0.2849 | 0.3295 |  0.2105|  <u>0.01616</u> |  5300 |  __0.016__ |
|ItemAsymkNN |  0.0642 |  0.2374 |  0.2838 |  0.2035 |  0.005823 |  9836 |  <u>0.0104</u> |
|MostPop |  0,0118 | 0.04305 |  0.05797 |  0.04223 |  0.001054 | 12478 |  0.00 |


## Repository Structure

- `data`: contains the data used for the recommendation algorithms
    - `evaluation`: contains the evaluation files and parameter settings
    - `processed`: contains the filtered dataset
    - `raw`: contains the raw un-filtered dataset
- `notebooks`:
    - `DataObservation.ipynb`: initial observation of the raw dataset
    - `Evaluation.ipynb`: evaluates the best parameter settings
    - `EvaluationParameterSearch.ipynb`: evaluates the parameter search
    - `ParameterSearchItem.ipynb`: the parameter search logic for the item-based algorithms
    - `ParameterSearchMF.ipynb`: the parameter search logic for the alternativ least squares algorithm
    - `ParameterSearchUser.ipynb`: the parameter search logic for the user-based algorithms 
- `src`: Enthält Entwicklungscode zum Backend inkl. separater `README.md` mit Erläuterungen
    - `Helper.py`: contains often used functions
    - `Metrics.py`: contains the entire metric and evluation logic
    - `MfAlgorithms.py`: contains the matrix factorization recommender
    - `NeighborhoodAlgorithms.py`: contains the neighborhood-based recommender
    - `NonPersonalizedAlgorithms.py`: contains the non-personalized recommender
- `CONDA_Requirements.txt`: contains all installed libraries generated via `conda list -e CONDA_Requirements.txt`
- `PIP_Requirements.txt`: contains all installed libraries generated via `pip3 freeze > PIP_Requirements.txt`
- `Seminar_Paper.pdf`: the corresponding paper _"Non-Deep Learning Collaborative Filtering Techniques for Implicit Feedback Data in Music Recommender Systems"_

#

## Setup

### Download and store the Dataset

The dataset can be found in the [Google Drive](https://drive.google.com/drive/folders/1PI544RBO7bsEMeUBa-pnzRYXKXgHcfJG?usp=sharing). Store the containing files as follows:

- `user_item_interaction_RAW_ANONYMIZED.txt` -> `data/raw`
- `user_item_interaction_FILTERED_ANONYMIZED.txt` -> `data/processed`

### Setup Virtual Environment

#### Setup with Conda (recommended)

Create virtual environment with predefined libraries:

        conda create --name <env> --file requirements.txt

#### Setup with Pip

0. Use python version between 3.8 and 3.9 (3.9.19 used and recommended)

1.  Create virtual environment in project directory (`ResearchSeminarMusicRecommender2024`):

        python3 -m venv .venv

2.  Activate virtual environment:

- on Mac / Linux:

        source .venv/bin/activate

- on Windows:

        .venv\Scripts\activate

3. Upgrade pip

        pip3 install --upgrade pip

4.  Install Packages

        pip install -r PIP_Requirements.txt


### Export requirements.txt

Open the terminal, activate the environment, and run: 

#### For Conda

        conda list -e > requirements.txt

#### For Pip

        pip list --format=freeze > requirements.txt
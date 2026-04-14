# my-encoding-model-project/README.md

# Encoding Model for Brain Responses

This project implements an encoding model to predict brain responses based on various inputs, specifically focusing on the relationship between memory confidence scores and brain activity. The analysis aims to determine whether higher confidence correlates with increased activity in specific brain regions.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Scripts Overview](#scripts-overview)
- [Data](#data)
- [License](#license)

## Introduction
The encoding model developed in this project utilizes intracranial EEG and fMRI data to analyze brain responses during memory tasks. By incorporating confidence scores related to memory, the model aims to provide insights into the neural mechanisms underlying memory retrieval and recognition.

## Installation
To set up the project environment, follow these steps:

1. Install Anaconda or Miniconda by following the instructions on the official [website](https://www.anaconda.com/).
2. Clone/download this repository to your local machine and navigate to the root directory of the repository in your terminal:
   ```bash
   git clone https://github.com/yourusername/my-encoding-model-project
   cd my-encoding-model-project
   ```
3. Create a new environment using the provided `make_env.yml` file:
   ```bash
   conda env create --file make_env.yml
   ```
4. Activate the environment:
   ```bash
   conda activate your_environment_name
   ```

## Usage
The main functionality is implemented in the Jupyter Notebook located in the `notebook` directory. Open the notebook `encoding_model_brain_responses.ipynb` to explore the encoding model, run analyses, and visualize results.

## Scripts Overview
The project includes several scripts for data processing and analysis:
- **ephys_qc**: Scripts for quality control and visualization of electrophysiological data.
- **fmri_qc**: Scripts for assessing fMRI data quality and performing inter-subject correlation analyses.
- **gen_table1_subj_info.py**: Generates a summary table of patient demographics and recording information.

## Data
Data used in this project includes intracranial EEG recordings and fMRI data. Please refer to the respective sections in the `README.md` of the main repository for instructions on how to download and preprocess the data.

## License
This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.
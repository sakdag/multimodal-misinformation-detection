# Multimodal Misinformation Detection

This project is part of a PhD class in applied NLP, focusing on multimodal misinformation detection. The repository includes scripts for data downloading, preprocessing, and building classifiers to detect misinformation using multimodal data.

## Table of Contents
- [Project Overview](#project-overview)
- [Environment Setup](#environment-setup)
- [Installing Dependencies](#installing-dependencies)
- [Downloading the Data](#downloading-the-data)

## Project Overview

This project aims to detect misinformation in multimodal data. The repository is organized to allow for data downloading, preprocessing, and experimentation with different classifiers. Key components include:
- A data downloader script to fetch the required dataset.
- A modular setup to add and train multiple classifiers.
- Utility functions and preprocessing scripts to prepare data for training.

## Environment Setup

### 1. Create a Conda Environment
Ensure [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) is installed on your system. Use the following command to create a Conda environment with Python 3.12:

```bash
conda create -n misinfo-detect python=3.12
```

### 2. Activate the Environment
```bash
conda activate misinfo-detect
```

### 3. Install Dependencies
Install the required dependencies using the following command:

```bash 
pip install -r requirements.txt
```

### 4. Downloading the Data
To download the dataset, run the following command:

```bash
python src/data_loader/download_data.py
```

Then data will appear in the `data` directory.

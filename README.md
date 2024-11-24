# Multimodal Misinformation Detection

This project is part of a PhD class in applied NLP, focusing on multimodal misinformation detection. The repository includes scripts for data downloading, preprocessing, and building classifiers to detect misinformation using multimodal data.

## Table of Contents
- [Project Overview](#project-overview)
- [Environment Setup](#environment-setup)
- [Installing Dependencies](#installing-dependencies)
- [Add Formatter](#add-formatter)
- [Downloading the Data](#downloading-the-data)

## Project Overview

This project aims to detect misinformation in multimodal data. 

## Environment Setup

### 1. Create a Conda Environment
Ensure [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) is installed on your system. Use the following command to create a Conda environment with Python 3.12:

```bash
conda create -n <your_env_name> python=3.12
```

### 2. Activate the Environment
```bash
conda activate <your_env_name>
```

## Install Dependencies
Install the required dependencies using the following command:

```bash 
pip install -r requirements.txt
```

## Add Formatter
Add black formatter to the project by completing the 
[editor integration using this guide](https://black.readthedocs.io/en/stable/integrations/editors.html). Note that 
the formatter is already installed as part of Install Dependencies step.

## Downloading the Data
To download the dataset, run the following command:

```bash
python -m src.data_loader.download_data
python -m src.data_loader.download_images
```

Then data will appear in the `data/raw/factify` directory.

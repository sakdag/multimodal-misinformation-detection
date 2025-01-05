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

## Downloading the Data
To download the dataset, run the following command:

```bash
python -m src.data_loader.download_data
```

Then data will appear in the `data/raw/factify` directory. 
The zip file will be downloaded to `data/raw/factify/factify_data.zip` and 
extracted to `data/raw/factify/extracted`. You should see `train.csv` and `test.csv` separately.

## Downloading the images
Using the image links in the CSVs, to download all images, run below.

```bash
python -m src.data_loader.download_images
```

Then images will appear in the `data/raw/factify/extracted/images` directory.

## Preprocessing the data
To apply the preprocessing scripts, run the following commands:

```bash
python -m src.preprocess.preprocess
```

Then preprocessed data will appear in the `data/processed/` directory. 
You should see `train.csv` and `test.csv` separately.

## Enriching claim and evidence text using image captioning
To run image captioning, run the following commands:

```bash
python -m src.preprocess.caption
```

This will go over the preprocessed data and enrich the claim and evidence text with image captions. 
The new columns will appear in the `data/processed/` directory. 
You should see `train_enriched.csv` and `test_enriched.csv` separately.

## Running the streamlit demo app
To run the streamlit demo app, run the following command:

```bash
python -m streamlit run src/demo/app.py
```

This will open a new tab in your browser with the demo app. 
You should be able to enter multimodal claims and play around with the trained model and the pipeline.
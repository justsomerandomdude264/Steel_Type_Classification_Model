# Steel Type Classification Model

## Overview

This machine learning model utilizes a Random Forest Classifier from scikit-learn to classify types of steel. It takes the following input features:

- Electricity Used (kWh)
- Lagging Current Power (kVarh)
- Leading Current Power (kVarh)
- CO2 Emissions (tCO2)
- Lagging Current Power Factor
- Leading Current Power Factor
- NSM (Normalized Slope of the Main Load)
- Week Status
- Day of the Week

The model predicts one of the three classes:

- Light Load
- Medium Load
- Maximum Load

## Dataset

The information in this dataset was gathered from DAEWOO Steel Co. Ltd in Gwangyang, South Korea. It produces several types of coils, steel plates, and iron plates. The dataset contains information on electricity consumption, which is held in a cloud-based system.

The energy consumption data of the industry is sourced from the website of the Korea Electric Power Corporation (pccs.kepco.go.kr). Perspectives on daily, monthly, and annual data are calculated and shown.

**Link**-https://www.kaggle.com/datasets/nimapourmoradi/steel-dataset/data

## Usage

This repository contains 2 files:

1. **data_setup.py**: This script takes the path to the data, converts it into a pandas dataframe, then removes unnecessary columns and renames some columns for a better view of the data. It preprocesses the data by, for example, converting text data into numbers and creating train and test sets.

2. **fit.py**: This script fits the model to the preprocessed data and prints the metrics, including accuracy, recall, precision, and F1 score.

Additionally, there is a CSV file containing the dataset.

## What I Learned

Through this project, I learned that in tabular and structured data, machine learning methods like ensemble models could easily outperform neural networks and deep learning approaches. After testing XGBoost, GradientBoostClassifier, SVMs, and initially neural networks, it became evident that ensemble methods, specifically the Random Forest Classifier, performed the best in this specific use case.

From this project, I also learned the basics of classification using PyTorch, even though it was not used in my final conclusion. Furthermore, I gained an appreciation for the power of simplicity in machine learning models like ensemble methods.

# Classifier for Process Yield Prediction

## Project Overview

This project builds a machine learning pipeline to analyze sensor data from a manufacturing process and predict the Pass/Fail status of each run.  
It performs data cleaning, EDA, handles class imbalance using SMOTE, and trains multiple classifiers to find the best model.

## Features

- Data Cleaning & Preprocessing
  - Fills missing values in numerical columns with mean and categorical with mode.
  - Drops non-informative columns like 'Time'.

- Exploratory Data Analysis (EDA)
  - Histograms, boxplots, violin plots for distribution analysis.
  - Correlation heatmap to detect relationships among sensor readings.

- Handling Imbalance
  - Balances dataset with SMOTE (Synthetic Minority Over-sampling Technique).

- Model Building
  - Random Forest with GridSearchCV for hyperparameter tuning.
  - Support Vector Machine with kernel and C parameter tuning.
  - Naive Bayes as baseline.

- Evaluation
  - Prints classification reports (precision, recall, F1-score).
  - Displays accuracy comparison across models.

- Model Export
  - Saves the best model as 'best_model.pkl'.

## Project Structure

- main.py : Main script with all steps from preprocessing to model training
- signal-data.csv : Sample dataset with sensor measurements
- best_model.pkl : Pickled best model after training
- README.md : Project documentation
- requirements.txt : List of dependencies




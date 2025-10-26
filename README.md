# ELEVATE_LABS---LTV-Prediction
Customer Lifetime Value (CLV) Prediction

This project builds a machine learning model to predict Customer Lifetime Value (CLV) using an insurance marketing dataset. It applies automated data preprocessing, feature engineering, and model optimization with Random Forest and XGBoost regressors.                 

## Dataset:
LTV (Marketing-Customer-Value-Analysis).csv

 - Contains customer and insurance policy information including:
 - Demographics (State, Education, Gender, Marital Status)
 - Policy data (Coverage, Vehicle Class, Policy Type)
 - Financials (Income, Monthly Premium Auto, Total Claim Amount)

Target: Customer Lifetime Value

# Requirements

Run in Google Colab or install dependencies locally:

pip install pandas numpy scikit-learn matplotlib seaborn joblib xgboost

# Model Training

Upload the dataset and run the main script: python clv_model.py

The script:

    Loads and cleans data
    
    Trains Random Forest and optionally XGBoost
    
    Evaluates both models
    
    Saves the best-performing model
    
    Exports predictions and visualizations

# Outputs

- Predicted CLV: Stored in outputs/clvpred.csv

- Segmented Customers: Stored in outputs/clvseg.csv

Plots:

    clvhist.png — CLV value distribution
    
    clvsegplot.png — Boxplot of CLV across segments

CLV Segmentation

Predicted CLV values are split into three groups based on quantiles:

    Segment	Description	CLV Range
    Low	Bottom 33%	≤ Q1
    Medium	Middle 33%	Q1–Q2
    High	Top 33%	> Q2

## Step 1: Prepare Your Environment

Install all dependencies:

pip install pandas numpy scikit-learn matplotlib seaborn joblib xgboost


## Step 2: Get the Dataset

Download the dataset:

File name:
Marketing-Customer-Value-Analysis.csv

Upload the CSV file

## Step 2: Write the Code

Cleans the dataset

Automatically identifies numeric and categorical features

Builds a preprocessing pipeline (imputation, scaling, encoding)

Splits the data into training and testing sets

Trains a Random Forest model with grid search

Saves the best model and predictions

Segments customers by predicted CLV and exports reports and plots

## Step 3: Run the Model

Open a new Colab notebook

Upload the CSV file

Run the cell — training and evaluation will start

Outputs will be saved in /content/models/ and /content/outputs/

## Step 4: Check the Outputs

After training completes:

clvpred.csv              # Predicted CLV for all customers
clvseg.csv               # Segmented customers (Low, Medium, High)
clvhist.png              # Distribution of predicted CLV
clvsegplot.png           # Boxplot of CLV segments

CSV files: Open in Excel or pandas to inspect predicted CLV and segments

Plots: View the images for distribution and segment visualization

Model: You can reload and reuse the trained model later with joblib.load()

## Step 5: Interpret the Results

Metric	Description

MAE (Mean Absolute Error)	Average prediction error (lower = better)

RMSE (Root Mean Squared Error)	Penalizes larger errors (lower = better)

R² (R-squared)	Indicates how much variance in CLV is explained by the model (higher = better)

Example output:
RF MAE: 1560.231  RMSE: 2380.510  R2: 0.84

## Step 6: Explore Customer Segments

In outputs/clvseg.csv, 

     Customer	PredCLV	Segment
     CUST001	2300.12	Low
     CUST002	7900.55	High
     CUST003	5200.33	Medium

To:

Prioritize high-value customers

Target retention campaigns

Identify churn risks in low-value segments

## Step 7: Visualize Results

View the generated plots:

outputs/clvhist.png → Histogram of predicted CLV values

outputs/clvsegplot.png → Boxplot comparing CLV across segments

These help visualize the distribution and separation between customer groups.

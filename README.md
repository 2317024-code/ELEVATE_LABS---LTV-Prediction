# ELEVATE_LABS---LTV-Prediction
Customer Lifetime Value (CLV) Prediction

This project builds a machine learning model to predict Customer Lifetime Value (CLV) using an insurance marketing dataset. It applies automated data preprocessing, feature engineering, and model optimization with Random Forest and XGBoost regressors.

# Features

End-to-end ML pipeline for CLV prediction

Automatic handling of missing values and categorical encoding

Model tuning via GridSearchCV

Optional XGBoost integration

Model saving (.joblib) and full-dataset prediction output

CLV segmentation (Low / Medium / High)

Automatic generation of plots and CSV outputs

Project Structure
.
├── models/
│   └── bestclv.joblib              
├── outputs/
│   ├── clvpred.csv                 
│   ├── clvseg.csv                  
│   ├── clvhist.png                
│   └── clvsegplot.png              
├── WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv
├── clv_model.py                   
└── README.md                      

# Dataset

## Dataset:
LTV (Marketing-Customer-Value-Analysis).csv

Contains customer and insurance policy information including:

Demographics (State, Education, Gender, Marital Status)

Policy data (Coverage, Vehicle Class, Policy Type)

Financials (Income, Monthly Premium Auto, Total Claim Amount)

Target: Customer Lifetime Value

# Requirements

Run in Google Colab or install dependencies locally:

pip install pandas numpy scikit-learn matplotlib seaborn joblib xgboost

# Model Training

Upload the dataset to /content (if using Colab).

Run the main script:

python clv_model.py


The script:

Loads and cleans data

Trains Random Forest and optionally XGBoost

Evaluates both models

Saves the best-performing model

Exports predictions and visualizations

# Outputs

Predicted CLV: Stored in outputs/clvpred.csv

Segmented Customers: Stored in outputs/clvseg.csv

Plots:

clvhist.png — CLV value distribution

clvsegplot.png — Boxplot of CLV across segments

# Example Metrics
Model	MAE	RMSE	R²
Random Forest	~1600	~2400	0.83
XGBoost	~1500	~2300	0.85

(Values depend on training conditions)

# CLV Segmentation

Predicted CLV values are split into three groups based on quantiles:

Segment	Description	CLV Range
Low	Bottom 33%	≤ Q1
Medium	Middle 33%	Q1–Q2
High	Top 33%	> Q2
Step 1: Prepare Your Environment

You can build this project either in Google Colab or locally using Python 3.10+.

# Required Libraries

Install all dependencies:

pip install pandas numpy scikit-learn matplotlib seaborn joblib xgboost


If using Google Colab, these libraries are pre-installed (except xgboost, which you can add with !pip install xgboost if needed).

## Step 2: Get the Dataset

Download the dataset:

File name:
WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv

You can find it here:
Customer Value Analysis Dataset on Kaggle

Upload the CSV file to:

/content/ folder (if using Google Colab)

Your project directory (if using VS Code or local Jupyter)

## Step 3: Create the Project Folder

Create a new folder structure:

clv_project/
├── clv_model.py
├── WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv
├── README.md
└── requirements.txt


Later, when you run the script, it will automatically generate:

models/
outputs/

## Step 4: Write the Code

Create a file named clv_model.py and paste the cleaned script (the one without underscores and comments):

# shortened version from previous answer

This code:

Loads and cleans the dataset

Automatically identifies numeric and categorical features

Builds a preprocessing pipeline (imputation, scaling, encoding)

Splits the data into training and testing sets

Trains a Random Forest model with grid search

Optionally trains XGBoost if installed

Saves the best model and predictions

Segments customers by predicted CLV and exports reports and plots

## Step 5: Run the Model
🔹 Option A: In Google Colab

Open a new Colab notebook

Upload the CSV file

Paste the code from clv_model.py into a new cell

Run the cell — training and evaluation will start

Outputs will be saved in /content/models/ and /content/outputs/

🔹 Option B: Locally

Run the script in your terminal:

python clv_model.py

🪜 Step 6: Check the Outputs

After training completes, you’ll have:

models/
 └── bestclv.joblib           # Saved trained model

outputs/
 ├── clvpred.csv              # Predicted CLV for all customers
 ├── clvseg.csv               # Segmented customers (Low, Medium, High)
 ├── clvhist.png              # Distribution of predicted CLV
 └── clvsegplot.png           # Boxplot of CLV segments

# Verify the outputs:

CSV files: Open in Excel or pandas to inspect predicted CLV and segments

Plots: View the images for distribution and segment visualization

Model: You can reload and reuse the trained model later with joblib.load()

## Step 7: Interpret the Results
Metric	Description
MAE (Mean Absolute Error)	Average prediction error (lower = better)
RMSE (Root Mean Squared Error)	Penalizes larger errors (lower = better)
R² (R-squared)	Indicates how much variance in CLV is explained by the model (higher = better)

Example output:

RF MAE: 1560.231  RMSE: 2380.510  R2: 0.84


The script will automatically choose the best-performing model between Random Forest and XGBoost.

## Step 8: Explore Customer Segments

In outputs/clvseg.csv, you’ll find each customer’s predicted CLV and their assigned segment:

Customer	PredCLV	Segment
CUST001	2300.12	Low
CUST002	7900.55	High
CUST003	5200.33	Medium

Use this to:

Prioritize high-value customers

Target retention campaigns

Identify churn risks in low-value segments

## Step 9: Visualize Results

View the generated plots:

outputs/clvhist.png → Histogram of predicted CLV values

outputs/clvsegplot.png → Boxplot comparing CLV across segments

These help visualize the distribution and separation between customer groups.

## Step 10: Deploy or Extend

Once trained, you can:

Reload the model in a web app or dashboard

Integrate with Flask or Streamlit to create an interactive CLV predictor

Add new features like tenure, loyalty score, or churn risk

Automate retraining using scheduled scripts or Airflow

Example for reusing the model:

import joblib, pandas as pd
m = joblib.load('models/bestclv.joblib')
newdata = pd.read_csv('new_customers.csv')
preds = m.predict(newdata)

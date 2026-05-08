# Carbon Emission Estimator for Supply Chains

## Overview
This project predicts carbon emissions in supply chains using Machine Learning.  
It helps organizations analyze sustainability and operational factors to estimate carbon emissions and improve green logistics.

---

## Features
- Predict carbon emissions in kg CO2e
- Interactive Streamlit web application
- Multiple regression models
- Sustainability recommendations
- Emission comparison charts
- Supply chain optimization insights

---

## Technologies Used
- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Joblib

---

## Dataset
Dataset File:
`SCM_Dataset_Updated_with_Green_Logistics.xlsx`

Dataset includes:
- Supplier Count
- Lead Time
- Energy Consumption
- Renewable Energy Usage
- Recycling Rate
- Green Packaging Usage
- Operational Efficiency
- Customer Satisfaction
- Supply Chain Risk

Target Variable:
- Carbon Emissions (kg CO2e)

---

## Machine Learning Models
The following models are used:
- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor

The best-performing model is automatically selected.

---

## Installation

### Install dependencies
```bash
pip install -r requirements.txt
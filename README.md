# Unicorn Startup Valuation Analysis

This repository contains an in-depth analysis of unicorn startup valuations, highlighting key insights, trends, and model predictions. The project utilizes Python for data preprocessing, exploratory data analysis (EDA), feature engineering, and machine learning model implementation.

---

## Project Structure

The repository is organized as follows:

- **`data/`**: Contains the unicorn startup dataset used for the analysis.
- **`notebooks/`**: Includes Jupyter notebooks for data exploration, feature engineering, and model training.
- **`src/`**: Python scripts for preprocessing, visualization, and model implementation.
- **`output/`**: Generated visualizations, reports, and model outputs.
- **`README.md`**: Overview of the project and instructions for usage.

---

## Objectives

1. Analyze trends in the unicorn startup ecosystem across countries, industries, and time.
2. Identify key features influencing valuations.
3. Develop predictive models to estimate startup valuations.

---

## Dataset Overview

The dataset contains information on unicorn startups, including:
- Company Name
- Valuation ($B)
- Date Joined
- Country
- City
- Industry
- Key Investors

### Key Data Cleaning Steps:
- Standardized and transformed valuation figures into numeric format.
- Extracted temporal features such as "Year" and "Month" from the date.
- One-hot encoded categorical variables for compatibility with machine learning models.

---

## Key Visualizations

### 1. **Top 10 Countries by Total Valuation**
   - **Purpose**: Highlights the geographic trends in startup valuations.
   - **Location in Report**: Included in the "Exploratory Data Analysis" section.

   ![Top 10 Countries Visualization](output/top_10_countries.png)

### 2. **Top 10 Feature Importances**
   - **Purpose**: Shows the key factors influencing unicorn startup valuations, based on the Random Forest model.
   - **Location in Report**: Placed in the "Feature Importance" section.

   ![Feature Importance Visualization](output/feature_importance.png)

---

## Modeling Approach

### 1. Linear Regression
- Baseline model to assess relationships between features and valuation.
- Performance:
  - **Mean Squared Error (MSE)**: 9.18 × 10^26
  - **R-squared**: -1.17 × 10^26

### 2. Random Forest Regressor
- Captures non-linear feature interactions and provides feature importance rankings.
- Performance:
  - **Mean Squared Error (MSE)**: 32.30
  - **R-squared**: -3.13

---

## Key Insights

1. **Geographic Trends**
   - The United States leads in total unicorn startup valuation, followed by China and the United Kingdom.

2. **Temporal Patterns**
   - Founding year and month strongly influence valuations, showcasing the importance of market timing.

3. **Industry Focus**
   - Fintech, e-commerce, and AI are dominant industries, reflecting high investor interest.

4. **Feature Importance**
   - Temporal features (Year, Month) and country/industry categories drive valuation predictions.

---

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import joblib

# Load the dataset (replace 'unicorn_startups.csv' with your actual file)
data = pd.read_csv('value.csv')

# Display dataset information
print("Dataset Information:")
print(data.info())

# Data cleaning
data['Valuation ($B)'] = data['Valuation ($B)'].str.replace('$', '').astype(float)
data['Date Joined'] = pd.to_datetime(data['Date Joined'])
data.dropna(subset=['Investors'], inplace=True)

# Feature engineering
data['Year'] = data['Date Joined'].dt.year
data['Month'] = data['Date Joined'].dt.month

# Handle outliers in Valuation
q1 = data['Valuation ($B)'].quantile(0.25)
q3 = data['Valuation ($B)'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
data = data[(data['Valuation ($B)'] >= lower_bound) & (data['Valuation ($B)'] <= upper_bound)]

# Apply log transformation to Valuation
data['Log Valuation ($B)'] = np.log1p(data['Valuation ($B)'])

# Exploratory Data Analysis (EDA) - Top 10 countries by total valuation
country_valuation = data.groupby('Country')['Valuation ($B)'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=country_valuation.index, y=country_valuation.values, palette='viridis')
plt.title("Top 10 Countries by Total Valuation", fontsize=16)
plt.xlabel("Country", fontsize=12)
plt.ylabel("Total Valuation ($B)", fontsize=12)
plt.xticks(rotation=45)
plt.show()

# Prepare data for modeling
X = pd.get_dummies(data[['Year', 'Month', 'Country', 'Industry']], drop_first=True)
y = data['Log Valuation ($B)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Random Forest Model with Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='r2')
grid_search.fit(X_train, y_train)
rf_model = grid_search.best_estimator_
y_pred_rf = rf_model.predict(X_test)

# Model evaluation
print("Linear Regression:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_lr)}")
print(f"R-squared: {r2_score(y_test, y_pred_lr)}")

print("\nRandom Forest:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_rf)}")
print(f"R-squared: {r2_score(y_test, y_pred_rf)}")

# Feature importance visualization (Random Forest)
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
feature_importances.nlargest(10).plot(kind='barh', color='skyblue')
plt.title("Top 10 Feature Importances", fontsize=16)
plt.xlabel("Importance", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.show()

# Save the Random Forest model
joblib.dump(rf_model, 'random_forest_model.pkl')

# 🏠 House Price Prediction — Regression Model Comparison

A machine learning project for predicting house prices using the **Ames Housing / House Prices - Advanced Regression Techniques** dataset. This project compares the performance of several regression algorithms to determine the model with the best prediction accuracy.

---

# 📖 Overview

This project aims to build a house price prediction model (**SalePrice**) based on 79 property features, such as lot area, material quality, year built, garage condition, and many others.

Three regression algorithms were compared to determine which one provides the most accurate predictions:

- Lars (Least Angle Regression)
- Linear Regression
- Gradient Boosting Regressor

The best-performing model was then saved in both **`.joblib`** and **`.pkl`** formats so it can be reused without retraining.

---

# 🔄 Workflow

The **`House_Price.ipynb`** notebook follows the pipeline below:

### 1. Data Cleaning & Missing Value Handling

- Columns with fewer than **1,000 missing values** were filled using the **median** (for numerical features) or the **mode** (for categorical features).
- Columns with **1,000 or more missing values** were removed.

### 2. Outlier Removal

- The **Interquartile Range (IQR)** method was used to detect and remove outliers from numerical features.

### 3. Feature Scaling

- Numerical features were standardized using **StandardScaler**.

### 4. Categorical Data Encoding

- Two encoding approaches were explored:
  - One-Hot Encoding
  - Label Encoding

### 5. Exploratory Data Analysis (EDA)

- Feature distribution visualization
- Outlier visualization using boxplots
- Correlation matrix analysis to identify features most correlated with **SalePrice**

### 6. Train-Test Split

- The dataset was split into **80% training** and **20% testing** using **`random_state=1`**.

### 7. Model Training & Evaluation

- Three regression models were trained and evaluated using:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R² Score

### 8. Model Saving

- The best-performing model (**Gradient Boosting Regressor**) was saved as:
  - `gbr_model.joblib`
  - `gbr_model.pkl`

---

# 📊 Model Performance Comparison

| Model | MAE | MSE | R² Score |
|:------|----:|----:|---------:|
| Lars | 39,256.54 | 2,341,737,000 | 0.0433 |
| Linear Regression | 13,106.33 | 292,029,600 | 0.8807 |
| **Gradient Boosting Regressor** | **11,969.57** | **269,872,100** | **0.8897** |

The **Gradient Boosting Regressor** achieved the best performance, with an **R² score of approximately 0.89** and the lowest **Mean Absolute Error (MAE)** among the three models. Therefore, it was selected as the final model for this project.

---

# 🛠 Technologies Used

- Python 3
- Pandas & NumPy — Data manipulation and analysis
- Scikit-learn — Data preprocessing, model training, and evaluation
- Seaborn & Matplotlib — Data visualization
- Joblib & Pickle — Model serialization and saving

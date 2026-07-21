# House Price Prediction — Regression Model Comparison

Proyek machine learning untuk memprediksi harga rumah menggunakan dataset Ames Housing / House Prices - Advanced Regression Techniques. Proyek ini membandingkan performa beberapa algoritma regresi untuk menentukan model dengan akurasi prediksi terbaik.

# Overview 
Proyek ini bertujuan untuk membangun model prediksi harga rumah (SalePrice) berdasarkan 79 fitur properti seperti luas tanah, kualitas material, tahun dibangun, kondisi garasi, dan lain-lain. Tiga algoritma regresi dibandingkan untuk melihat mana yang memberikan hasil prediksi paling akurat:

Lars (Least Angle Regression)
Linear Regression
Gradient Boosting Regressor

Model terbaik kemudian disimpan dalam format .joblib dan .pkl agar dapat digunakan kembali tanpa perlu melatih ulang.

# Alur Kerja

Notebook House_Price.ipynb menjalankan pipeline berikut secara berurutan:

1. Data Cleaning & Missing Value Handling Kolom dengan missing value < 1000 diisi menggunakan median (numerik) atau modus (kategorikal). Kolom dengan missing value ≥ 1000 dibuang.
2. Outlier Removal Menggunakan metode IQR (Interquartile Range) untuk mendeteksi dan membuang data outlier pada fitur numerik.
3. Feature Scaling Standarisasi fitur numerik menggunakan StandardScaler.
4. Encoding Data Kategorikal Eksplorasi dua pendekatan: One-Hot Encoding dan Label Encoding untuk mengubah data kategorikal menjadi numerik.
5. Exploratory Data Analysis (EDA) Visualisasi distribusi fitur, boxplot outlier, serta correlation matrix untuk melihat fitur yang paling berkorelasi dengan SalePrice.
6. Train-Test Split Data dibagi 80% training dan 20% testing (random_state=1).
7. Model Training & Evaluation Melatih dan mengevaluasi 3 model regresi menggunakan metrik MAE, MSE, dan R².
8. Model Saving Model dengan performa terbaik (Gradient Boosting Regressor) disimpan ke gbr_model.joblib dan gbr_model.pkl.


# Hasil & Perbandingan Model
Model	MAE	MSE	R²
| Model | MAE | MSE | R² Score |
|:------|----:|----:|---------:|
| Lars | 39,256.54 | 2,341,737,000 | 0.0433 |
| Linear Regression | 13,106.33 | 292,029,600 | 0.8807 |
| **Gradient Boosting Regressor** | **11,969.57** | **269,872,100** | **0.8897** |

Gradient Boosting Regressor menjadi model dengan performa terbaik, dengan nilai R² ≈ 0.89 dan error (MAE) paling rendah dibanding dua model lainnya. Model inilah yang digunakan sebagai model final proyek.


# Teknologi yang Digunakan
Python 3
Pandas & NumPy — manipulasi dan analisis data
Scikit-learn — preprocessing, training, dan evaluasi model
Seaborn & Matplotlib — visualisasi data
Joblib & Pickle — penyimpanan model

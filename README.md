# SC2 Detection Using Deep Learning

A Flask-based machine learning and deep learning application for SC2 detection using multi-omic and demographic data. The application provides data analysis, classification, model performance evaluation, and prediction functionality through a web interface.

## 📌 Project Overview

This project implements a web-based machine learning system for SC2 detection.

The application supports:

* Dataset upload through a web interface
* Exploratory data analysis (EDA)
* Data preprocessing
* Multiple machine learning classification models
* Deep learning-based models
* Model performance evaluation
* Prediction on uploaded CSV data
* Visualization of analysis and model results
* SQLite-based application data management

## ✨ Features

- Upload CSV datasets through the web application
- Perform exploratory data analysis (EDA)
- Preprocess uploaded data
- Compare multiple machine learning classifiers
- Use deep learning models for SC2 detection
- Generate predictions from uploaded datasets
- Evaluate model performance
- Visualize data and model results
- Store and process uploaded files through the Flask application
## 🛠️ Technologies Used

### Programming & Framework

* Python
* Flask

### Data Processing & Visualization

* Pandas
* NumPy
* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* Random Forest
* Support Vector Machine
* K-Nearest Neighbors
* Logistic Regression
* Decision Tree
* Gradient Boosting
* AdaBoost

### Deep Learning

* TensorFlow
* Keras
* Dense Neural Networks
* Batch Normalization
* Dropout

### Database & Utilities

* SQLite
* Pickle
* Joblib

## 📂 Project Structure

```text
sc2-detection-deep-learning/
│
├── Dataset/
│   ├── Data.csv
│   └── testdata.csv
│
├── model/
│   ├── adaboost.pkl
│   ├── decision tree.pkl
│   ├── encoders.pkl
│   ├── gradient boosting.pkl
│   ├── k-nearest neighbors.pkl
│   ├── logistic regression.pkl
│   ├── proposed_DBN_SNN_ORLC.pkl
│   ├── proposed_hybrid_dl.pkl
│   ├── random forest.pkl
│   ├── results.json
│   └── support vector machine.pkl
│
├── static/
│   └── css/
│       └── style.css
│
├── templates/
│   ├── base.html
│   ├── classification.html
│   ├── eda.html
│   ├── performance.html
│   ├── predict.html
│   └── predict_results.html
│
├── uploads/
│   └── .gitkeep
│
├── app.py
├── orlc.py
├── run3120.bat
├── .gitignore
└── README.md
```

## ⚙️ Machine Learning Models

The application includes several classification models:

* Random Forest
* Support Vector Machine
* K-Nearest Neighbors
* Logistic Regression
* Decision Tree
* Gradient Boosting
* AdaBoost

The project also includes deep learning models and trained model files for prediction and comparison.

## 🔬 Data Processing

The application uses Pandas and NumPy for data processing and Scikit-learn utilities for preprocessing, including:

* Label Encoding
* Feature Scaling
* Train/Test Splitting

## 📊 Model Evaluation

The application uses evaluation metrics including:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

## 🌐 Web Application

The Flask application provides web pages for:

* Data analysis
* Classification
* Model performance
* Prediction
* Prediction results

Uploaded CSV files are processed through the application and prediction results can be generated dynamically.

## 🚀 Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/shaikhasheera/sc2-detection-deep-learning.git
cd sc2-detection-deep-learning
```

### 2. Install the required dependencies

Install the Python packages used by the project, including Flask, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib, and TensorFlow.

### 3. Run the Flask application

```bash
python app.py
```

The application can then be accessed through the local Flask server.

## 👩‍💻 Author

**Shaik Hasheera**

Computer Science Graduate | Java Full Stack Developer

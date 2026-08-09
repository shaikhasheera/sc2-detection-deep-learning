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

## 🛠️ Tech Stack

| Category            | Technologies        |
| ------------------- | ------------------- |
| Backend             | Python, Flask       |
| Machine Learning    | Scikit-learn        |
| Deep Learning       | TensorFlow, Keras   |
| Data Processing     | Pandas, NumPy       |
| Visualization       | Matplotlib, Seaborn |
| Database            | SQLite              |
| Frontend            | HTML, CSS           |
| Model Serialization | Pickle, Joblib      |

## 🎯 Project Objective

The objective of this project is to develop a web-based system for SC2 detection using multi-omic and demographic data.

The system combines data preprocessing, exploratory data analysis, traditional machine learning algorithms, and deep learning approaches to support classification and prediction through a Flask-based web application.

## 👩‍💻 My Contribution

* Developed the Flask-based web application for SC2 detection.
* Implemented data upload and preprocessing workflows.
* Integrated multiple machine learning classification models.
* Integrated deep learning models using TensorFlow and Keras.
* Implemented model prediction and evaluation functionality.
* Developed web pages for EDA, classification, performance analysis, and prediction.
* Organized trained models and supporting project files for application use.

## ⭐ Project Highlights

* Web-based SC2 detection application built with Flask
* Supports both machine learning and deep learning approaches
* Includes multiple classification algorithms for model comparison
* Provides exploratory data analysis and visualization
* Supports CSV dataset upload and prediction
* Includes model performance evaluation using multiple metrics
* Uses trained model files for prediction through the web application

## ✨ Features

* Upload CSV datasets through the web application
* Perform exploratory data analysis (EDA)
* Preprocess uploaded data
* Compare multiple machine learning classifiers
* Use deep learning models for SC2 detection
* Generate predictions from uploaded datasets
* Evaluate model performance
* Visualize data and model results
* Store and process uploaded files through the Flask application

## 🔄 Project Workflow

```text
CSV Dataset
     │
     ▼
Data Upload
     │
     ▼
Data Preprocessing
     │
     ▼
Exploratory Data Analysis (EDA)
     │
     ▼
Feature Processing
     │
     ├───────────────┬────────────────┐
     ▼               ▼                ▼
Machine Learning   Deep Learning   Model Evaluation
Models             Models
     │               │                │
     └───────────────┴────────────────┘
                     │
                     ▼
                SC2 Prediction
                     │
                     ▼
              Prediction Results
```

The Flask web application provides the interface for uploading datasets, performing analysis, running classification models, evaluating model performance, and generating predictions.

## 🤖 Models Used

### Machine Learning Models

The application includes the following machine learning classifiers:

* Random Forest
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)
* Logistic Regression
* Decision Tree
* Gradient Boosting
* AdaBoost

### Deep Learning Models

The project also includes deep learning approaches using TensorFlow and Keras, including:

* Proposed DBN-SNN model
* Proposed Hybrid Deep Learning model

The trained models are stored in the `model/` directory and are used by the application for prediction and model comparison.

## 🔬 Data Processing

The application uses Pandas and NumPy for data processing and Scikit-learn utilities for preprocessing, including:

* Label Encoding
* Feature Scaling
* Train/Test Splitting

## 📊 Model Evaluation

The application evaluates classification models using multiple performance metrics:

| Metric                | Purpose                                                          |
| --------------------- | ---------------------------------------------------------------- |
| Accuracy              | Measures the overall proportion of correct predictions           |
| Precision             | Measures the proportion of positive predictions that are correct |
| Recall                | Measures how effectively positive cases are identified           |
| F1 Score              | Combines precision and recall into a single metric               |
| Confusion Matrix      | Shows the distribution of correct and incorrect predictions      |
| Classification Report | Provides detailed classification performance metrics             |

These metrics are used to analyze and compare the performance of the implemented models.

## 🌐 Application Modules

The Flask web application is organized into multiple modules:

| Module             | Purpose                                              |
| ------------------ | ---------------------------------------------------- |
| Home               | Provides the main interface for the application      |
| EDA                | Performs exploratory data analysis and visualization |
| Classification     | Runs and compares classification models              |
| Performance        | Displays model evaluation results                    |
| Prediction         | Allows users to submit data for prediction           |
| Prediction Results | Displays the generated prediction results            |

The application uses Flask templates with HTML and CSS to provide the web interface.

## 📂 Dataset

The project includes dataset files used for model development and testing:

```text
Dataset/
├── Data.csv
└── testdata.csv
```

The application also supports CSV file uploads through the web interface for processing and prediction.

## 📁 Project Structure

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

## 📋 Directory Overview

| Directory/File | Description                                                     |
| -------------- | --------------------------------------------------------------- |
| `Dataset/`     | Contains the project datasets used for processing and testing   |
| `model/`       | Contains trained machine learning and deep learning model files |
| `static/`      | Contains CSS and other static web assets                        |
| `templates/`   | Contains HTML templates used by the Flask application           |
| `uploads/`     | Stores files uploaded or generated by the application           |
| `app.py`       | Main Flask application and web application logic                |
| `orlc.py`      | Contains the ORLC implementation used by the project            |
| `run3120.bat`  | Local Windows batch script used during development              |
| `.gitignore`   | Specifies files and folders that should not be committed to Git |

## 🚀 Running the Project

### Prerequisites

* Python 3.x
* pip
* A Python virtual environment is recommended

### 1. Clone the repository

```bash
git clone https://github.com/shaikhasheera/sc2-detection-deep-learning.git
cd sc2-detection-deep-learning
```

### 2. Create and activate a virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

Install the Python libraries required by the project, including Flask, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Joblib, and TensorFlow.

> A `requirements.txt` file will be added after verifying the exact dependencies and versions used by the project.

### 4. Run the application

```bash
python app.py
```

The Flask application will start on the local development server. Open the URL displayed in the terminal to access the application.

## 👩‍💻 Author

**Shaik Hasheera**

Computer Science Graduate | Java Full Stack Developer

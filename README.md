# SC2 Detection Using Deep Learning

A deep learning-based approach for SC2 detection using multi-omic and demographic data.

## 📌 Project Overview

This project focuses on detecting SC2 using a neural-network-based approach applied to multi-omic and demographic data.

The project explores data preprocessing, model development, training, and prediction to identify SC2-related patterns from the available dataset.

## 🎯 Objective

* Process and prepare the dataset for machine learning.
* Use relevant multi-omic and demographic features.
* Develop a deep learning-based detection model.
* Train and evaluate the model using the prepared dataset.
* Generate predictions for unseen test data.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Deep Learning / Neural Networks
* Jupyter Notebook

## 📊 Dataset

The dataset was processed and cleaned before model training.

* Original records: **702**
* Cleaned records: **234**
* Training records: **187**
* Testing records: **47**

## 🧠 Model

The project uses a **Dense Batch Normalization-Serial Neural Network (DBN-SNN)** approach for SC2 detection.

The trained model is used to generate predictions based on the selected input features.

## 📁 Project Structure

```text
sc2-detection-deep-learning/
│
├── app.py
├── data.csv
├── Test data.csv
├── proposed_DBN_SNN_ORLC.pkl
├── adaboost.pkl
├── decision_tree.pkl
├── gradient_boosting.pkl
├── k-nearest_neighbors.pkl
├── logistic_regression.pkl
├── encoders.pkl
└── README.md
```

## 🚀 How to Run

1. Clone the repository.
2. Install the required Python libraries.
3. Make sure the required dataset and model files are available.
4. Run the application using:

```bash
python app.py
```

## 📈 Models Compared

The project includes trained models for comparison, including:

* Proposed DBN-SNN model
* AdaBoost
* Decision Tree
* Gradient Boosting
* K-Nearest Neighbors
* Logistic Regression

## 👩‍💻 Author

**Shaik Hasheera**

Computer Science Graduate | Java Full Stack Developer

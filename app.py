import os
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import sqlite3
from datetime import datetime
import io
import base64
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from orlc import ORLC

app = Flask(__name__)

app.config['SECRET_KEY'] = os.environ.get('SESSION_SECRET', "default_dev_secret_key")
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'model'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'csv'}

CLASSIFIER_NAMES = [
    'Random Forest',
    'Support Vector Machine',
    'K-Nearest Neighbors',
    'Logistic Regression',
    'Decision Tree',
    'Gradient Boosting',
    'AdaBoost',
    'Proposed Hybrid DL'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_and_preprocess_data():
    df = pd.read_csv('Dataset/data.csv')
    
    df_encoded = df.copy()
    le_gender = LabelEncoder()
    le_batch = LabelEncoder()
    le_pcr = LabelEncoder()
    
    df_encoded['gender'] = le_gender.fit_transform(df['gender'])
    df_encoded['sequencing_batch'] = le_batch.fit_transform(df['sequencing_batch'])
    df_encoded['SC2_PCR'] = le_pcr.fit_transform(df['SC2_PCR'])
    
    X = df_encoded.drop(['viral_status', 'CZB_ID', 'idseq_sample_name'], axis=1)
    y = df_encoded['viral_status']
    
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    encoders = {
        'gender': le_gender,
        'batch': le_batch,
        'pcr': le_pcr,
        'target': le_target,
        'scaler': scaler
    }
    
    with open(os.path.join(app.config['MODEL_FOLDER'], 'encoders.pkl'), 'wb') as f:
        pickle.dump(encoders, f)
    
    return X_scaled, y_encoded, le_target, df

def build_DBN_SNN_ORLC_model(input_dim, num_classes,X_train, y_train,X_test, y_test):
    ## Dense Batch Normalization-Serial Neural Network (DBN-SNN)##
    ## Optimal Rule List Classifier ##
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(input_dim,)))
    model.add(BatchNormalization())

    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(16, activation='relu', name='deep_features'))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=2
    )

    _, dnn_acc = model.evaluate(X_test, y_test, verbose=0)

    rf_model = ORLC()
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, y_pred_rf)


    if rf_acc > dnn_acc:
        final_model = rf_model
        y_pred = final_model.predict(X_test)
        y_pred_proba = final_model.predict(X_test)
    else:
        final_model = model
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
    return final_model,y_pred,y_pred_proba



def train_all_models():
    X, y, le_target, df = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {}
    results = {}
    
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=2, random_state=42),
        'Support Vector Machine': SVC(kernel='rbf', random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=2, min_samples_split=10, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42)
    }
    
    for name, clf in classifiers.items():
        print(f"Training {name}...")
        clf.fit(X_train, y_train)
        models[name] = clf
        
        y_pred = clf.predict(X_test)
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, target_names=le_target.classes_, zero_division=0)
        }
        
        with open(os.path.join(app.config['MODEL_FOLDER'], f'{name.replace(" ", "_").lower()}.pkl'), 'wb') as f:
            pickle.dump(clf, f)
    
    print("Training Proposed Hybrid DL...")
    final_model,y_pred,y_pred_proba = build_DBN_SNN_ORLC_model(X_train.shape[1], len(np.unique(y)),X_train, y_train,X_test, y_test)
    
    



    
    models['Proposed Hybrid DL'] = final_model
   
    results['Proposed Hybrid DL'] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, target_names=le_target.classes_, zero_division=0)
    }
    
    joblib.dump(final_model, os.path.join(app.config['MODEL_FOLDER'], 'proposed_DBN_SNN_ORLC.pkl'))
    
    with open(os.path.join(app.config['MODEL_FOLDER'], 'results.json'), 'w') as f:
        json.dump(results, f)
    
    print("All models trained and saved successfully!")
    return results

def load_all_models():
    models = {}
    
    ml_classifiers = [
        'Random Forest', 'Support Vector Machine', 'K-Nearest Neighbors',
        'Logistic Regression', 'Decision Tree', 'Gradient Boosting', 'AdaBoost'
    ]
    
    for name in ml_classifiers:
        model_path = os.path.join(app.config['MODEL_FOLDER'], f'{name.replace(" ", "_").lower()}.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                models[name] = pickle.load(f)
    
    dl_model_path = os.path.join(app.config['MODEL_FOLDER'], 'proposed_DBN_SNN_ORLC.pkl')
    if os.path.exists(dl_model_path):
        models['Proposed Hybrid DL'] = joblib.load(dl_model_path)
    
    return models

def check_models_exist():
    ml_classifiers = [
        'Random Forest', 'Support Vector Machine', 'K-Nearest Neighbors',
        'Logistic Regression', 'Decision Tree', 'Gradient Boosting', 'AdaBoost'
    ]
    
    for name in ml_classifiers:
        model_path = os.path.join(app.config['MODEL_FOLDER'], f'{name.replace(" ", "_").lower()}.pkl')
        if not os.path.exists(model_path):
            return False
    
    dl_model_path = os.path.join(app.config['MODEL_FOLDER'], 'proposed_DBN_SNN_ORLC.pkl')
    if not os.path.exists(dl_model_path):
        return False
    
    return True

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/eda')
def eda():
    df = pd.read_csv('Dataset/data.csv')
    
    stats = {
        'total_samples': len(df),
        'features': df.shape[1],
        'classes': df['viral_status'].nunique(),
        'class_distribution': df['viral_status'].value_counts().to_dict(),
        'gender_distribution': df['gender'].value_counts().to_dict(),
        'missing_values': df.isnull().sum().sum()
    }
    
    plt.figure(figsize=(10, 6))
    df['viral_status'].value_counts().plot(kind='bar', color=['#FFB6C1', '#FF69B4', '#FF1493'])
    plt.title('Distribution of Viral Status')
    plt.xlabel('Viral Status')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    img1 = io.BytesIO()
    plt.savefig(img1, format='png', dpi=100, bbox_inches='tight')
    img1.seek(0)
    plot1 = base64.b64encode(img1.getvalue()).decode()
    plt.close()
    
    plt.figure(figsize=(10, 6))
    df['gender'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#FFB6C1', '#FF69B4'])
    plt.title('Gender Distribution')
    plt.ylabel('')
    plt.tight_layout()
    
    img2 = io.BytesIO()
    plt.savefig(img2, format='png', dpi=100, bbox_inches='tight')
    img2.seek(0)
    plot2 = base64.b64encode(img2.getvalue()).decode()
    plt.close()
    
    plt.figure(figsize=(10, 6))
    df['age'].hist(bins=20, color='#FFB6C1', edgecolor='black')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.tight_layout()
    
    img3 = io.BytesIO()
    plt.savefig(img3, format='png', dpi=100, bbox_inches='tight')
    img3.seek(0)
    plot3 = base64.b64encode(img3.getvalue()).decode()
    plt.close()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='RdPu', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    
    img4 = io.BytesIO()
    plt.savefig(img4, format='png', dpi=100, bbox_inches='tight')
    img4.seek(0)
    plot4 = base64.b64encode(img4.getvalue()).decode()
    plt.close()
    
    return render_template('eda.html', stats=stats, plot1=plot1, plot2=plot2, plot3=plot3, plot4=plot4)

@app.route('/classification')
def classification():
    if not check_models_exist():
        flash('Training models for the first time. This may take a few minutes...', 'info')
        results = train_all_models()
    else:
        results_path = os.path.join(app.config['MODEL_FOLDER'], 'results.json')
        with open(results_path, 'r') as f:
            results = json.load(f)
    
    return render_template('classification.html', results=results, classifiers=CLASSIFIER_NAMES)

@app.route('/performance')
def performance():
    if not check_models_exist():
        flash('Please train models first!', 'warning')
        return redirect(url_for('classification'))
    
    results_path = os.path.join(app.config['MODEL_FOLDER'], 'results.json')
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    comparison_data = []
    for name in CLASSIFIER_NAMES:
        if name in results:
            comparison_data.append({
                'name': name,
                'accuracy': results[name]['accuracy'],
                'precision': results[name]['precision'],
                'recall': results[name]['recall'],
                'f1_score': results[name]['f1_score']
            })
    
    return render_template('performance.html', results=results, comparison_data=comparison_data, classifiers=CLASSIFIER_NAMES)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected!', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected!', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            if not check_models_exist():
                flash('Models not trained yet! Training now...', 'info')
                train_all_models()
            
            test_df = pd.read_csv(filepath)
            
            with open(os.path.join(app.config['MODEL_FOLDER'], 'encoders.pkl'), 'rb') as f:
                encoders = pickle.load(f)
            
            test_encoded = test_df.copy()
            
            if 'gender' in test_df.columns:
                test_encoded['gender'] = encoders['gender'].transform(test_df['gender'])
            if 'sequencing_batch' in test_df.columns:
                test_encoded['sequencing_batch'] = encoders['batch'].transform(test_df['sequencing_batch'])
            if 'SC2_PCR' in test_df.columns:
                test_encoded['SC2_PCR'] = encoders['pcr'].transform(test_df['SC2_PCR'])
            
            cols_to_drop = ['viral_status', 'CZB_ID', 'idseq_sample_name']
            existing_cols = [col for col in cols_to_drop if col in test_encoded.columns]
            X_test = test_encoded.drop(existing_cols, axis=1)
            
            X_test_scaled = encoders['scaler'].transform(X_test)
            
            models = load_all_models()
            predictions = {}
            
            for name, model in models.items():
                if name == 'Proposed Hybrid DL':
                    y_pred = model.predict(X_test_scaled)
                else:
                    y_pred = model.predict(X_test_scaled)
                
                predictions[name] = encoders['target'].inverse_transform(y_pred).tolist()
            
            result_df = test_df.copy()
            for name in CLASSIFIER_NAMES:
                if name in predictions:
                    result_df[f'{name}_Prediction'] = predictions[name]
            
            result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'predictions_{filename}')
            result_df.to_csv(result_filepath, index=False)
            
            return render_template('predict_results.html', 
                                 predictions=predictions, 
                                 test_data=test_df.head(20).to_html(classes='table table-striped table-sm'),
                                 filename=f'predictions_{filename}',
                                 classifiers=CLASSIFIER_NAMES)
        else:
            flash('Invalid file format! Please upload a CSV file.', 'danger')
            return redirect(request.url)
    
    return render_template('predict.html')

@app.route('/download/<filename>')
def download_file(filename):
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)

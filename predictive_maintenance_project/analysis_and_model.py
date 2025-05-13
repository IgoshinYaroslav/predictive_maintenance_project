import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import re

def clean_column_names(df):
    """Очищает названия столбцов от недопустимых символов"""
    df.columns = [re.sub(r'[\[\]<>]', '', str(col)) for col in df.columns]
    return df

def preprocess_data(data):
    """Предварительная обработка данных"""
    # Удаление ненужных столбцов
    cols_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    data = data.drop(columns=[col for col in cols_to_drop if col in data.columns])
    
    # Кодирование категориальных признаков
    if 'Type' in data.columns:
        data['Type'] = LabelEncoder().fit_transform(data['Type'])
    
    # Очистка названий столбцов
    data = clean_column_names(data)
    
    # Масштабирование числовых признаков
    numerical_features = ['Air temperature K', 'Process temperature K', 
                         'Rotational speed rpm', 'Torque Nm', 'Tool wear min']
    numerical_features = [col for col in numerical_features if col in data.columns]
    
    if numerical_features:
        scaler = StandardScaler()
        data[numerical_features] = scaler.fit_transform(data[numerical_features])
    
    return data

def train_models(X_train, y_train):
    """Обучение моделей"""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    return models

def evaluate_models(models, X_test, y_test):
    """Оценка качества моделей"""
    results = {}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Метрики
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }
        
        # ROC-кривая
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax.plot(fpr, tpr, label=f'{name} (AUC = {results[name]["roc_auc"]:.2f})')
    
    # Настройка графика ROC-кривых
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend()
    
    return results, fig

def analysis_and_model_page():
    st.title("Анализ данных и модель")
    
    # Загрузка данных
    uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            data = preprocess_data(data)
            
            # Разделение данных
            if 'Machine failure' in data.columns:
                X = data.drop(columns=['Machine failure'])
                y = data['Machine failure']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Обучение моделей
                models = train_models(X_train, y_train)
                
                # Оценка моделей
                results, roc_fig = evaluate_models(models, X_test, y_test)
                
                # Визуализация результатов
                st.header("Результаты обучения моделей")
                
                for name, metrics in results.items():
                    st.subheader(name)
                    st.write(f"Accuracy: {metrics['accuracy']:.2f}")
                    st.write(f"ROC-AUC: {metrics['roc_auc']:.2f}")
                    
                    # Матрица ошибок
                    st.write("Confusion Matrix:")
                    fig, ax = plt.subplots()
                    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
                    st.pyplot(fig)
                
                # ROC-кривые
                st.subheader("ROC Curves")
                st.pyplot(roc_fig)
                
                # Интерфейс для предсказания
                st.header("Предсказание по новым данным")
                with st.form("prediction_form"):
                    st.write("Введите значения признаков:")
                    
                    input_data = {}
                    for col in X.columns:
                        if col == 'Type':
                            input_data[col] = st.selectbox(col, [0, 1, 2])
                        else:
                            input_data[col] = st.number_input(col)
                    
                    submit_button = st.form_submit_button("Предсказать")
                    
                    if submit_button:
                        input_df = pd.DataFrame([input_data])
                        # Применяем масштабирование, если оно было
                        if hasattr(preprocess_data, 'scaler'):
                            num_cols = [col for col in input_df.columns if col in preprocess_data.scaler.feature_names_in_]
                            input_df[num_cols] = preprocess_data.scaler.transform(input_df[num_cols])
                        
                        best_model = models["Random Forest"]
                        prediction = best_model.predict(input_df)
                        proba = best_model.predict_proba(input_df)[0][1]
                        
                        st.success(f"Предсказание: {'Отказ' if prediction[0] == 1 else 'Нет отказа'}")
                        st.info(f"Вероятность отказа: {proba:.2f}")
            
            else:
                st.error("Столбец 'Machine failure' не найден в данных!")
        
        except Exception as e:
            st.error(f"Ошибка при обработке данных: {str(e)}")
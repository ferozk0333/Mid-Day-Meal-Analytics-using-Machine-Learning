# Training ML models
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import joblib



hygiene = pd.read_csv("data/processed/hygiene_processed.csv")
meals = pd.read_csv("data/processed/meals_processed.csv")
attendance = pd.read_csv('data/processed/attendance_processed.csv')
meals = pd.read_csv('data/processed/meals_processed.csv')
schools = pd.read_csv('data/processed/schools_processed.csv')
students = pd.read_csv('data/processed/students_processed.csv')

def quality_and_hygiene_predictor_classifier():
    # Merge on school_id and date
    combined_data = pd.merge(meals, hygiene, on=['school_id', 'date'], how='inner')
    # We must also ensure that no missing values remain in key features
    combined_data.fillna(method='ffill', inplace=True)

    # Let's first define the target variable - this is the dependent variable
    combined_data['target'] = (
        (combined_data['meal_quality_score'] >= 7.5) &
        (combined_data['hygiene_score'] >= 7.5)
    ).astype(int)

    # Now, let's build the model

    # Train TEst split
    # Here, I will select input or independent features
    X = combined_data[['cooking_temperature', 'serving_temperature', 'meal_quality_score',
                   'meal_wastage', 'calories_per_meal', 'cooking_serving_diff',
                   'menu_Chapati & Vegetables', 'menu_Idli & Sambar', 'menu_Khichdi',
                   'menu_Pulao', 'menu_Rice & Dal']]
    y = combined_data['target']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    #Model
    # Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, min_samples_split=2) # Based on outcomes of GridSearchCV from Jupyter notebook

    # Perform stratified k-fold cross-validation
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='f1')
    print(f"Average F1-Score from CV: {np.mean(cv_scores):.4f}")

    # Train the model on the full training set
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test)

    # Evaluate performance
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save the model
    joblib.dump(rf_model, 'data/models/quality_model.pkl')
    print("Classification model saved.")

def anomaly_detector_using_clustering():
    # Aggregate students data
    students_agg = students.groupby('school_id').agg({
    'BMI': 'mean',
    'family_income': 'mean',
    'academic_performance_High': 'sum'  # Count of high-performing students
    }).rename(columns={
    'BMI': 'avg_BMI',
    'family_income': 'avg_family_income',
    'academic_performance_High': 'num_high_performers'
    })

    # Aggregate meals data
    meals_agg = meals.groupby('school_id').agg({
    'calories_per_meal': 'mean',
    'meal_quality_score': 'mean'
    }).rename(columns={
    'calories_per_meal': 'avg_calories_per_meal',
    'meal_quality_score': 'avg_meal_quality_score'
    })

    # Aggregate attendence data
    attendance_agg = attendance.groupby('school_id').agg({
    'attendance_rate': 'mean'
    }).rename(columns={
    'attendance_rate': 'avg_attendance_rate'
    })

    # Merge aggregated datasets
    school_data = students_agg.merge(meals_agg, on='school_id').merge(attendance_agg, on='school_id')

    # Let's us also add daily-level features (attendance + meals)
    daily_data = pd.merge(attendance, meals, on=['school_id', 'date'], how='inner')

    # Feattures for clustering
    clustering_data = daily_data[['attendance_rate', 'meal_wastage', 'cooking_serving_diff', 'calories_per_meal','school_id']]
    clustering_data = pd.merge(clustering_data, school_data, on='school_id')

    # Handling NaNs
    mean_value = clustering_data['cooking_serving_diff'].mean()
    clustering_data['cooking_serving_diff'] = clustering_data['cooking_serving_diff'].fillna(mean_value)

    # Standardize features
    scaler = StandardScaler()
    clustering_data_scaled = scaler.fit_transform(clustering_data)

    dbscan = DBSCAN(eps=2.5, min_samples=5) 
    anomaly_labels_1 = dbscan.fit_predict(clustering_data_scaled)

    # Ading anomaly labels to the dataset
    clustering_data['cluster'] = anomaly_labels_1

    # Analysis
    # Group data by cluster and calculate mean values
    cluster_summary = clustering_data.groupby('cluster').mean()
    print(cluster_summary)

    joblib.dump(dbscan, 'data/models/dbscan_model.pkl')
    print("Cljuster model saved")




anomaly_detector_using_clustering()
quality_and_hygiene_predictor_classifier()
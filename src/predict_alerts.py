# Alert generation using trained models
import pandas as pd
import numpy as np

hygiene_data = pd.read_csv("data/processed/hygiene_processed.csv")
meals_data = pd.read_csv("data/processed/meals_processed.csv")
attendance_data = pd.read_csv('data/processed/attendance_processed.csv')
meals_data = pd.read_csv('data/processed/meals_processed.csv')
schools_data = pd.read_csv('data/processed/schools_processed.csv')
students_data = pd.read_csv('data/processed/students_processed.csv')

def generate_alerts():
    alerts = []

    # Focus on Attendance Alerts
    attendance_data['absenteeism_percent'] = 100 - (attendance_data['attendance_rate'] * 100)
    high_absenteeism = attendance_data[attendance_data['absenteeism_percent'] > 50]
    for index, row in high_absenteeism.iterrows():
        alerts.append({
            'school_id': row['school_id'],
            'alert_type': 'High Absenteeism',
            'details': f"Absenteeism is {row['absenteeism_percent']:.2f}% on {row['date']}"
        })

    # Hygiene Alerts
    low_hygiene_scores = hygiene_data[hygiene_data['hygiene_score'] < 7]
    for index, row in low_hygiene_scores.iterrows():
        alerts.append({
            'school_id': row['school_id'],
            'alert_type': 'Low Hygiene Score',
            'details': f"Hygiene score is {row['hygiene_score']} on {row['date']}"
        })

    # Meal Distribution Alerts
    leftovers_threshold = 20
    meals_data['leftovers_percent'] = (meals_data['meal_wastage'] / (meals_data['calories'] + 1e-9)) * 100
    high_leftovers = meals_data[meals_data['leftovers_percent'] > leftovers_threshold]
    for index, row in high_leftovers.iterrows():
        alerts.append({
            'school_id': row['school_id'],
            'alert_type': 'High Leftovers',
            'details': f"Meal leftovers are {row['leftovers_percent']:.2f}% on {row['date']}"
        })

    alerts = pd.DataFrame(alerts)
    alerts.to_csv("data/processed/alerts.csv", index=False)
    print("Success")

# alerts = generate_alerts()

# alerts.to_csv("data/processed/alerts.csv", index=False)
# print("Success")
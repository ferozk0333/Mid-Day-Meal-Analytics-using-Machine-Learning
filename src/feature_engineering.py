import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from datetime import datetime

def process_meals_table(meals_df):
    # is_wastage_high: Binary feature for meal wastage - THRESHOLD SET TO 8
    meals_df['is_wastage_high'] = (meals_df['meal_wastage'] > 8).astype(int)

    # cooking_serving_diff: Difference between cooking and serving temperature
    meals_df['cooking_serving_diff'] = meals_df['cooking_temperature'] - meals_df['serving_temperature']

    # calories_per_meal: Ratio of calories to meal quality score
    meals_df['calories_per_meal'] = meals_df['calories'] / meals_df['meal_quality_score']

    # Encode weather_conditions
    weather_encoder = LabelEncoder()
    meals_df['weather_conditions_encoded'] = weather_encoder.fit_transform(meals_df['weather_conditions'])

    # LEt's use OHE for menu
    menu_onehot = pd.get_dummies(meals_df['menu'], prefix='menu')
    meals_df = pd.concat([meals_df, menu_onehot], axis=1)

    return meals_df

def process_attendance_table(attendance_df):
    attendance_df['attendance_rate'] = attendance_df.groupby(['school_id', 'date'])['status'].transform('mean')
    attendance_df = attendance_df.sort_values(by='date', ascending=True).round(2)

    # week_of_year: Extract   week number out of date date
    attendance_df['week_of_year'] = pd.to_datetime(attendance_df['date']).dt.isocalendar().week
    # Set the week_of_year for January 1 explicitly to 1
    attendance_df.loc[attendance_df['date'] == "2021-01-01", 'week_of_year'] = 1

    return attendance_df

def process_students_table(students_df):
    # BMI
    students_df['BMI'] = students_df['weight_kg'] / ((students_df['height_cm'] / 100) ** 2)

    # Main logic - Family income divided by number_of_siblings + 1
    students_df['income_per_sibling'] = students_df['family_income'] / (students_df['number_of_siblings'] + 1)

    # Let's encode gender
    gender_encoder = LabelEncoder()
    students_df['gender_encoded'] = gender_encoder.fit_transform(students_df['gender'])

    # Again, let's encode academic_performance using one-hot encoding
    academic_onehot = pd.get_dummies(students_df['academic_performance'], prefix='academic_performance')
    students_df = pd.concat([students_df, academic_onehot], axis=1)

    # Let's do parent occupation now
    parent_encoder = LabelEncoder()
    students_df['parent_occupation_encoded'] = parent_encoder.fit_transform(students_df['parent_occupation'])
    # Encoding transport_mode
    transport_encoder = LabelEncoder()
    students_df['transport_mode_encoded'] = transport_encoder.fit_transform(students_df['transport_mode'])

    return students_df

def process_hygiene_audits_table(hygiene_df):
    hygiene_df['remarks'] = hygiene_df['remarks'].fillna('No Remarks')

    remarks_encoder = LabelEncoder()
    hygiene_df['remarks_encoded'] = remarks_encoder.fit_transform(hygiene_df['remarks'])

    return hygiene_df

# Example Usage
if __name__ == "__main__":
    # Load datasets
    meals_df = pd.read_csv('data/raw/meals.csv')
    attendance_df = pd.read_csv('data/raw/attendance.csv')
    students_df = pd.read_csv('data/raw/students.csv')
    hygiene_df = pd.read_csv('data/raw/hygiene_audits.csv')

    # Process datasets
    meals_processed = process_meals_table(meals_df)
    attendance_processed = process_attendance_table(attendance_df, total_students_by_school=None)  # Provide actual mapping if available
    students_processed = process_students_table(students_df)
    hygiene_processed = process_hygiene_audits_table(hygiene_df)

    # Save processed datasets
    meals_processed.to_csv('data/processed/meals_processed.csv', index=False)
    attendance_processed.to_csv('data/processed/attendance_processed.csv', index=False)
    students_processed.to_csv('data/processed/students_processed.csv', index=False)
    hygiene_processed.to_csv('data/processed/hygiene_processed.csv', index=False)
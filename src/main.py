import os, sys

import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db_connection import save_table_as_csv
from preprocess_data import preprocess_meals, preprocess_attendance, preprocess_students
from feature_engineering import process_attendance_table, process_meals_table, process_students_table, process_hygiene_audits_table


def fetch_data_from_db_and_save_raw_csv(tables, output_dir):
    # Save each table as a CSV
    for table in tables:
        save_table_as_csv(table, output_dir)

def main():

    # Definingg directories
    raw_dir = os.path.join("data", "raw")
    powerbi_dir = os.path.join("data", "powerbi_data")
    processed_dir = os.path.join("data", "processed")

    # Check if directories exist, else create
    os.makedirs(powerbi_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # Step1: Fetching Data from DB and saving raw CSV
    tables = ['students', 'attendance', 'schools', 'hygiene_audits', 'meals']    # List of tables to fetch
    fetch_data_from_db_and_save_raw_csv(tables, raw_dir)

    # Step2: Load datasets and preprocess tables to clean data
    meals_df = pd.read_csv(os.path.join(raw_dir, "meals.csv"))
    students_df = pd.read_csv(os.path.join(raw_dir, "students.csv"))
    attendance_df = pd.read_csv(os.path.join(raw_dir, "attendance.csv"))
    hygiene_df = pd.read_csv(os.path.join(raw_dir, "hygiene_audits.csv"))
    schools_df = pd.read_csv(os.path.join(raw_dir, "schools.csv"))

    meals_cleaned = preprocess_meals(meals_df)
    students_cleaned = preprocess_students(students_df)
    attendance_cleaned = preprocess_attendance(attendance_df)

    # Step3: Save a copy of cleaned data for PowerBI functionality
    meals_cleaned.to_csv(os.path.join(powerbi_dir, "meals_cleaned.csv"), index=False)
    students_cleaned.to_csv(os.path.join(powerbi_dir, "students_cleaned.csv"), index=False)
    attendance_cleaned.to_csv(os.path.join(powerbi_dir, "attendance_cleaned.csv"), index=False)
    hygiene_df.to_csv(os.path.join(powerbi_dir, "hygiene_cleaned.csv"), index=False)  
    schools_df.to_csv(os.path.join(powerbi_dir, "schools_cleaned.csv"), index=False)

    print("Success: Cleaned data saved.") # debuggin check, remove later

    # Step4: Feature Engineering and Encoding
    meals_features = process_meals_table(meals_cleaned)
    students_features = process_students_table(students_cleaned)
    attendance_features = process_attendance_table(attendance_cleaned)
    hygiene_features = process_hygiene_audits_table(hygiene_df)

    # Step5: Save processed files
    meals_features.to_csv(os.path.join(processed_dir, "meals_processed.csv"), index=False)
    students_features.to_csv(os.path.join(processed_dir, "students_processed.csv"), index=False)
    attendance_features.to_csv(os.path.join(processed_dir, "attendance_processed.csv"), index=False)
    hygiene_features.to_csv(os.path.join(processed_dir, "hygiene_processed.csv"), index=False)
    schools_df.to_csv(os.path.join(processed_dir, "schools_processed.csv"), index=False)

    print("Success: Feature-engineered data saved.") # remove tis later...degug check

if __name__ == "__main__":
    main()

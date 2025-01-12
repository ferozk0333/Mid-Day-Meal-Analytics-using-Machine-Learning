# Preprocessing and feature engineering


import pandas as pd
import numpy as np


def preprocess_meals(df):
    # Replace cooking_temperature that is negative with absolute and zero with same category mean of non zero vlues
    # Convert negative cooking temperatures to positive
    df['cooking_temperature'] = df['cooking_temperature'].apply(lambda x: -x if x < 0 else x)

    # Replace cooking temperatures of 0 with None
    df['cooking_temperature'] = df['cooking_temperature'].apply(lambda x: None if x == 0 else x)

    # Calculate mean cooking temperature for each menu where cooking_temperature > 0
    mean_temps = df[df['cooking_temperature'] > 0].groupby('menu')['cooking_temperature'].mean()

    # Replace None in cooking_temperature with the corresponding menu mean
    df['cooking_temperature'] = df.apply(
    lambda row: mean_temps[row['menu']] if row['cooking_temperature'] is None else row['cooking_temperature'],
    axis=1
    )
    
    # Handle mean_qiality_score
    df['meal_quality_score'] = df['meal_quality_score'].apply(lambda x: None if x < 0 else x)
    df['meal_quality_score'] = df['meal_quality_score'].fillna(df['meal_quality_score'].mean())
    df['meal_quality_score'] = df['meal_quality_score'].apply(lambda x: 10.0 if x > 10 else x)
    
    # Fill missing remarks with 'No Remarks'
    df['remarks'] = df['remarks'].fillna('No Remarks')
    return df



def preprocess_attendance(df):
    """Preprocess the attendance table."""
    # Fill missing status with 0 (Absent)
    df['status'] = df['status'].fillna(0)
    df['status'] = df['status'].astype(int)

    return df


def preprocess_students(df):
    # Convert grade from text to numerical
    grade_mapping = {
        "1st":1, "2nd":2, "3rd":3, "4th":4,"5th":5
    }
    df['grade'] = df['grade'].map(grade_mapping)
    
    # Handle outliers and invalid values in height
    df['height_cm'] = df['height_cm'].apply(lambda x: None if x<=0 or x>250 else x)
    df['height_cm'] = df['height_cm'].fillna(df['height_cm'].median())
    
    # Handle outliers and invalid values in weight
    df['weight_kg'] = df['weight_kg'].apply(lambda x: None if x <= 0 or x > 200 else x)
    df['weight_kg'] = df['weight_kg'].fillna(df['weight_kg'].median())
    
    # Replace negative or zero family_income with NaN and fill missing with median
    df['family_income'] = df['family_income'].apply(lambda x: -1*x if x < 0 else x)
    df['family_income'] = df['family_income'].fillna(df['family_income'].median())
    
    return df


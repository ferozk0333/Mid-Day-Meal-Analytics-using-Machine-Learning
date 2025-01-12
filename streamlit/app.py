# Main Streamlit app for analytics

import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict_alerts import generate_alerts

scaler = StandardScaler()

# Load models
quality_model = joblib.load('data/models/quality_model.pkl')
dbscan_model = joblib.load('data/models/dbscan_model.pkl')

# Ajusting width of the page
st.set_page_config(
    page_title="Mid-Day Meal Analytics",
    page_icon="🍽️",
    #layout="wide", # To fit full screen
)
# App Header
st.title("Mid-Day Meal Analytics")

st.header("Quality and Hygiene Prediction")

# Add a small description for the module
st.markdown("""
This feature predicts if food quality and hygiene meet standards. It combines meals and hygiene tables with school_id and date, using key features like meal quality, hygiene score, and more to classify meals as **Acceptable** or **Substandard**.
""")

uploaded_file = st.file_uploader("Upload CSV containing formatted columns as per GitHub description", type=["csv"])
if uploaded_file:
    meal_data = pd.read_csv(uploaded_file)
    # Let's do some predictions to test the function
    predictions = quality_model.predict(meal_data)
    # Map predictions to labels so as to improve readability
    label_mapping = {0: "Substandard", 1: "Acceptable"}
    meal_data['Prediction'] = pd.Series(predictions).map(label_mapping)

    # Display graphs
    st.write(meal_data)
    st.bar_chart(meal_data['Prediction'].value_counts())

st.header("Anomaly Detection")

# Add a description for the Anomaly Detection module
st.markdown("""
This feature identifies irregularities in attendance and meal distribution patterns by leveraging unsupervised machine learning to flag anomalies such as inflated attendance or mismatched meal counts.
""")

uploaded_file = st.file_uploader("Upload CSV containing desired columns as per GitHub description", type=["csv"])
if uploaded_file:
    anomaly_data = pd.read_csv(uploaded_file)
    
    # Select numerical columns to scale (modify based on your dataset)
    numerical_columns = anomaly_data.select_dtypes(include=['float64', 'int64']).columns
    st.write("Scaling these columns:", numerical_columns.tolist())
    
    # Fit the scaler on the data
    scaler.fit(anomaly_data[numerical_columns])
    
    # Scale the data
    scaled_data = scaler.transform(anomaly_data[numerical_columns])
    
    # Run the anomaly detection model
    anomaly_labels = dbscan_model.fit_predict(scaled_data)
    anomaly_data['Anomaly Flag'] = anomaly_labels

    # Display results
    st.write(anomaly_data)
    st.write("Anomaly Distribution:")
    st.bar_chart(anomaly_data['Anomaly Flag'].value_counts())

    st.header("Cluster-Wise Drill Down Analytics")

    if 'Anomaly Flag' in anomaly_data.columns:
        clusters = anomaly_data['Anomaly Flag'].unique()  # To avoid errors
        selected_cluster = st.selectbox("Select a Cluster for Summary:", clusters)
        cluster_data = anomaly_data[anomaly_data['Anomaly Flag'] == selected_cluster]

        # Let's calculate mean, median, and SDV
        summary_stats = cluster_data.describe().transpose()
        summary_stats['Median'] = cluster_data.median()  # Have to add separately

        st.write(f"Here are the descriptive statistics for {selected_cluster}:")
        st.dataframe(summary_stats)
    else:
        st.warning("No clusters detected :(")






# Here, we can embed our powerbi dashboard
st.header("PowerBI Visualization Dashboard")

url = "https://app.powerbi.com/view?r=eyJrIjoiYjc2NmRkOGQtZDYxYi00YTg2LTllNDctZWYwMTU1MTU0MDlkIiwidCI6ImY2YjZkZDViLWYwMmYtNDQxYS05OWEwLTE2MmFjNTA2MGJkMiIsImMiOjZ9"
st.components.v1.html(
    f"""
    <iframe title="MDM_Dashboard" width="900" height="400.25" src="https://app.powerbi.com/reportEmbed?reportId=b1cc65e1-407f-4fe7-85b7-24f9df98213f&autoAuth=true&ctid=f6b6dd5b-f02f-441a-99a0-162ac5060bd2" frameborder="0" allowFullScreen="true"></iframe>
    """,
    height=420,
)



if st.button("Generate Alert Logs"):
    # Generating Alerts
    generate_alerts()
    alert_logs = pd.read_csv("data/processed/alerts.csv") 
    st.write("Generated Alert Logs")
    st.dataframe(alert_logs)





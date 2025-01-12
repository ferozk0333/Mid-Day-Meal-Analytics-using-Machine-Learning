# Mid-Day Meal Analytics Using Machine Learning

The Mid-Day Meal Scheme is a government initiative in India that provides free, nutritious meals to school children to improve their health, encourage school attendance, and reduce dropout rates. However, inefficiencies in ground-level implementation lead to distribution malpractices.
This end-to-end solution integrates predictive analytics, anomaly detection, and interactive dashboards to ensure food quality and identify discrepancies in meal distribution.

---

## Features

### 1. Quality and Hygiene Prediction - Classification
- **Problem**: Predict if food quality and hygiene meet standards.
- **Approach**:
  - Supervised ML to classify data points.
  - **Input**: Parameters like temperature logs, ingredient ratings, and hygiene audit scores.
- **Output**: Classify meals as "Acceptable" or "Substandard."

### 2. Anomaly Detection in Attendance and Meal Distribution - Clustering
- **Problem**: Identify irregularities in attendance and meal delivery.
- **Approach**:
  - Unsupervised ML to make clusters.
  - **Input**: Daily attendance, meal counts, and school-level statistics.
- **Output**: Flag anomalies such as inflated attendance or mismatched meal counts.

### 3. PowerBI Dashboard
- **Visualize**:
  - DAX queries to derive new columns.
  - Summary statistics and drill-down Analytics joining multiple tables.

---

## Installation

### Prerequisites
- Python 3.8+
- Git

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/ferozk0333/Mid-Day-Meal-Analytics-using-Machine-Learning.git
   cd Mid-Day-Meal-Analytics
   ```
2. Create Virtual Environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
2. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run app:
   ```bash
   streamlit run main.py
   ```

---

### **Future Enhancement ideas**
- Adulteration detection by integrating sensors to detect liquid adulteration.
- Capturing meal distribution using CCTV to capture meal contents on the plate and student's face.

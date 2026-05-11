import pandas as pd
import numpy as np  # noqa: F401
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

st.markdown("""
    <style>

    .main {
        background-color: #0b0f19;
        color: white;
    }

    h1, h2, h3 {
        color: #00ffcc;
    }

    .stMetric {
        background-color: #111827;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #00ffcc;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    </style>
""", unsafe_allow_html=True)



st.markdown("""
    <style>
    .main {
        background-color: #0b0f19;
        color: white;
    }

    h1, h2, h3 {
        color: #00ffcc;
    }

    .stMetric {
        background-color: #111827;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #00ffcc;
    }

    .stDataFrame {
        border: 1px solid #00ffcc;
        border-radius: 10px;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Threat Intelligence Dashboard",
    layout="wide"
)

# ----------------------------
# Login Authentication
# ----------------------------

username = st.sidebar.text_input("Username")

password = st.sidebar.text_input(
    "Password",
    type="password"
)

if username != "admin" or password != "cyber123":
    st.warning("🔐 Please login to access dashboard")
    st.stop()

st.title("🛡 Threat Intelligence Dashboard")
st.markdown("Cybersecurity Threat Monitoring and Analysis System")

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("threat_data.csv")

# Convert Date Column
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = pd.to_datetime(df['Date'])

# ----------------------------
# Live Attack Counter
# ----------------------------

attack_total = int(df['Attack_Count'].sum())

st.markdown(
    f"""
    <h3 style='color:#ff4b4b;'>
    ⚡ Live Attack Counter: {attack_total}
    </h3>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Sidebar Filters
# ----------------------------
st.sidebar.header("Filter Threats")

selected_threat = st.sidebar.multiselect(
    "Select Threat Type",
    options=df['Threat_Type'].unique(),
    default=df['Threat_Type'].unique()
)

selected_country = st.sidebar.multiselect(
    "Select Country",
    options=df['Country'].unique(),
    default=df['Country'].unique()
)

filtered_df = df[
    (df['Threat_Type'].isin(selected_threat)) &
    (df['Country'].isin(selected_country))
]

# ----------------------------
# Display Dataset
# ----------------------------
st.subheader("Threat Dataset")
st.dataframe(filtered_df)
# ----------------------------
# Download Threat Report
# ----------------------------

csv = filtered_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Download Threat Report",
    data=csv,
    file_name='threat_report.csv',
    mime='text/csv'
)

# ----------------------------
# KPI Metrics
# ----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Threats", len(filtered_df))
col2.metric("Total Attacks", filtered_df['Attack_Count'].sum())
col3.metric(
    "Critical Threats",
    len(filtered_df[filtered_df['Severity'] == 'Critical'])
)
# ----------------------------
# Threat Alert System
# ----------------------------

critical_count = len(
    filtered_df[filtered_df['Severity'] == 'Critical']
)

high_count = len(
    filtered_df[filtered_df['Severity'] == 'High']
)

if critical_count > 5:
    st.error(
        f"🚨 ALERT: {critical_count} Critical Threats Detected!"
    )

elif high_count > 10:
    st.warning(
        f"⚠ High Threat Activity Detected: {high_count} High Severity Threats"
    )

else:
    st.success("✅ System Status Stable")
    # ----------------------------
# Threat Risk Score
# ----------------------------

total_attacks = filtered_df['Attack_Count'].sum()

if total_attacks > 4000:
    risk_score = "CRITICAL"
    risk_color = "red"

elif total_attacks > 2500:
    risk_score = "HIGH"
    risk_color = "orange"

elif total_attacks > 1200:
    risk_score = "MEDIUM"
    risk_color = "yellow"

else:
    risk_score = "LOW"
    risk_color = "green"

st.markdown(
    f"""
    <h2 style='color:{risk_color};'>
    Threat Risk Level: {risk_score}
    </h2>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Threat Type Distribution
# ----------------------------
st.subheader("Threat Type Distribution")

fig1 = px.pie(
    filtered_df,
    names='Threat_Type',
    title='Threat Type Percentage'
)

st.plotly_chart(fig1, use_container_width=True)

# ----------------------------
# Country-wise Attacks
# ----------------------------
st.subheader("Country-wise Attack Count")

country_attack = filtered_df.groupby(
    'Country'
)['Attack_Count'].sum().reset_index()

fig2 = px.bar(
    country_attack,
    x='Country',
    y='Attack_Count',
    color='Country',
    title='Country-wise Attacks'
)

st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# Severity Analysis
# ----------------------------
st.subheader("Severity Analysis")

severity_count = filtered_df['Severity'].value_counts().reset_index()
severity_count.columns = ['Severity', 'Count']

fig3 = px.bar(
    severity_count,
    x='Severity',
    y='Count',
    color='Severity',
    title='Threat Severity Analysis'
)

st.plotly_chart(fig3, use_container_width=True)

# ----------------------------
# Timeline Analysis
# ----------------------------
# ----------------------------
# Threat Forecasting
# ----------------------------

st.subheader("Threat Forecasting")

forecast_df = filtered_df.copy()

forecast_df = forecast_df.reset_index()

forecast_df['Index'] = forecast_df.index

X_forecast = forecast_df[['Index']]
y_forecast = forecast_df['Attack_Count']

forecast_model = LinearRegression()

forecast_model.fit(X_forecast, y_forecast)

future_index = [[len(forecast_df) + i] for i in range(5)]

future_predictions = forecast_model.predict(
    future_index
)

future_df = pd.DataFrame({
    'Future_Day': range(1, 6),
    'Predicted_Attacks': future_predictions
})

fig_forecast = px.line(
    future_df,
    x='Future_Day',
    y='Predicted_Attacks',
    markers=True,
    title='Future Threat Prediction'
)

st.plotly_chart(
    fig_forecast,
    use_container_width=True
)
st.subheader("Attack Timeline")

fig4 = px.line(
    filtered_df,
    x='Date',
    y='Attack_Count',
    color='Threat_Type',
    markers=True,
    title='Attack Trend Over Time'
)

st.plotly_chart(fig4, use_container_width=True)
# ----------------------------
# Global Threat Map
# ----------------------------

st.subheader("Global Cyber Threat Map")

country_attack_map = filtered_df.groupby(
    'Country'
)['Attack_Count'].sum().reset_index()

fig5 = px.choropleth(
    country_attack_map,
    locations='Country',
    locationmode='country names',
    color='Attack_Count',
    hover_name='Country',
    color_continuous_scale='Reds',
    title='Global Threat Distribution'
)

st.plotly_chart(fig5, use_container_width=True)

# AI Anomaly Detection
# ----------------------------
# AI Anomaly Detection
# ----------------------------

st.subheader("AI Threat Anomaly Detection")

anomaly_data = filtered_df[['Attack_Count']]

iso_model = IsolationForest(
    contamination=0.1,
    random_state=42
)

filtered_df['Anomaly'] = iso_model.fit_predict(
    anomaly_data
)

anomalies = filtered_df[
    filtered_df['Anomaly'] == -1
]

st.write("Detected Suspicious Threat Activities")

st.dataframe(
    anomalies[
        ['Date', 'Threat_Type',
         'Country', 'Attack_Count']
    ]
)

# Machine Learning Section
st.subheader("Threat Prediction Model")

# ----------------------------
# Machine Learning Section
# ----------------------------
st.subheader("Threat Prediction Model")

# Encode categorical columns
le_threat = LabelEncoder()
le_country = LabelEncoder()
le_severity = LabelEncoder()

ml_df = df.copy()

ml_df['Threat_Type'] = le_threat.fit_transform(
    ml_df['Threat_Type']
)

ml_df['Country'] = le_country.fit_transform(
    ml_df['Country']
)

ml_df['Severity'] = le_severity.fit_transform(
    ml_df['Severity']
)

# Features and Target
X = ml_df[['Threat_Type', 'Country', 'Attack_Count']]
y = ml_df['Severity']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
prediction = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, prediction)

st.success(f"Model Accuracy: {accuracy * 100:.2f}%")

# ----------------------------
# Prediction Input Section
# ----------------------------
st.subheader("Predict Threat Severity")

threat_input = st.selectbox(
    "Threat Type",
    df['Threat_Type'].unique()
)

country_input = st.selectbox(
    "Country",
    df['Country'].unique()
)

attack_input = st.number_input(
    "Attack Count",
    min_value=1,
    max_value=1000,
    value=100
)

if st.button("Predict Severity"):

    threat_encoded = le_threat.transform(
        [threat_input]
    )[0]

    country_encoded = le_country.transform(
        [country_input]
    )[0]

    result = model.predict([[
        threat_encoded,
        country_encoded,
        attack_input
    ]])

    severity_result = le_severity.inverse_transform(result)

    st.warning(
        f"Predicted Severity: {severity_result[0]}"
    )

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown(
    "Developed for Data Science Cybersecurity Project"
)
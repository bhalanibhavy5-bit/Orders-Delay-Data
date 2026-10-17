import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="Delivery Delay Prediction",
    page_icon="🚚",
    layout="wide"
)

# -------------------------
# CUSTOM CSS
# -------------------------

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
}

.stButton>button{
    background-color:#2563eb;
    color:white;
    width:100%;
    border-radius:10px;
    height:50px;
    font-size:18px;
}

.result-success{
    background:#14532d;
    padding:18px;
    border-radius:10px;
    color:white;
    font-size:20px;
    font-weight:bold;
}

.result-danger{
    background:#7f1d1d;
    padding:18px;
    border-radius:10px;
    color:white;
    font-size:20px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# LOAD DATASET
# -------------------------

df = pd.read_csv("delivery_delay.csv")

TARGET = "Delivery_Delay"

X = df.drop(columns=[TARGET])
y = df[TARGET]

X = pd.get_dummies(X)

feature_cols = X.columns

# -------------------------
# TRAIN MODEL
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", DecisionTreeClassifier(
        random_state=42
    ))
])

model.fit(X_train, y_train)

# -------------------------
# HEADER
# -------------------------

st.title("🚚 Delivery Delay Prediction App")

st.write("""
This application uses a Decision Tree model to predict whether a
delivery will be delayed based on operational parameters.
""")

st.markdown("---")

st.subheader("Enter Delivery Details")

# -------------------------
# INPUT FORM
# -------------------------

col1, col2 = st.columns(2)

with col1:

    delivery_distance = st.number_input(
        "Delivery Distance",
        min_value=0.0,
        value=20.0
    )

    traffic_congestion = st.slider(
        "Traffic Congestion Level (1-5)",
        1,
        5,
        3
    )

    weather_condition = st.slider(
        "Weather Condition Level (1-3)",
        1,
        3,
        1
    )

    delivery_slot = st.slider(
        "Delivery Slot Time (1-3)",
        1,
        3,
        2
    )

    driver_experience = st.number_input(
        "Driver Experience (Years)",
        min_value=0,
        value=5
    )

    num_stops = st.number_input(
        "Number Of Stops",
        min_value=0,
        value=3
    )

with col2:

    vehicle_age = st.number_input(
        "Vehicle Age (Years)",
        min_value=0,
        value=3
    )

    road_condition_score = st.slider(
        "Road Condition Score (1-5)",
        1,
        5,
        3
    )

    package_weight = st.number_input(
        "Package Weight (lbs)",
        min_value=0.0,
        value=15.0
    )

    fuel_efficiency = st.number_input(
        "Fuel Efficiency (mpg)",
        min_value=0.0,
        value=12.0
    )

    warehouse_processing_time = st.number_input(
        "Warehouse Processing Time (Minutes)",
        min_value=0,
        value=45
    )

st.write("")

# -------------------------
# PREDICT BUTTON
# -------------------------

if st.button("Predict Delay Status"):

    input_data = pd.DataFrame({

        "Delivery_Distance":[delivery_distance],
        "Traffic_Congestion":[traffic_congestion],
        "Weather_Condition":[weather_condition],
        "Delivery_Slot":[delivery_slot],
        "Driver_Experience":[driver_experience],
        "Num_Stops":[num_stops],
        "Vehicle_Age":[vehicle_age],
        "Road_Condition_Score":[road_condition_score],
        "Package_Weight":[package_weight],
        "Fuel_Efficiency":[fuel_efficiency],
        "Warehouse_Processing_Time":[warehouse_processing_time]

    })

    input_data = pd.get_dummies(input_data)

    input_data = input_data.reindex(
        columns=feature_cols,
        fill_value=0
    )

    prediction = model.predict(input_data)[0]

    st.markdown("## Prediction Result")

    if prediction == 0:

        st.markdown(
            """
            <div class="result-success">
            ✅ Predicted Status : No Delay
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="result-danger">
            🚚 Predicted Status : Delivery Delayed
            </div>
            """,
            unsafe_allow_html=True
        )

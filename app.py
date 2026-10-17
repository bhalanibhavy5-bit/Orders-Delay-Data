import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

# ----------------------------
# PAGE CONFIG
# ----------------------------

st.set_page_config(
    page_title="Delivery Delay Prediction",
    page_icon="🚚",
    layout="wide"
)

# ----------------------------
# LOAD DATA
# ----------------------------

df = pd.read_csv("delivery_delay.csv")

TARGET = "Delivery_Delay"

X = df.drop(columns=[TARGET])
y = df[TARGET]

X = pd.get_dummies(X)

feature_columns = X.columns

# ----------------------------
# TRAIN MODEL
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", DecisionTreeClassifier(random_state=42))
])

model.fit(X_train, y_train)

# ----------------------------
# TITLE
# ----------------------------

st.title("🚚 Delivery Delay Prediction App")

st.write(
    """
    This application uses a Decision Tree model to predict
    whether a delivery will be delayed based on key
    operational parameters.
    """
)

st.divider()

st.header("Enter Delivery Details")

col1, col2 = st.columns(2)

# ----------------------------
# LEFT SIDE
# ----------------------------

with col1:

    delivery_distance = st.number_input(
        "Delivery Distance",
        value=20.0
    )

    traffic = st.slider(
        "Traffic Congestion (1-5)",
        1,
        5,
        3
    )

    weather = st.slider(
        "Weather Condition (1-3)",
        1,
        3,
        1
    )

    delivery_slot = st.slider(
        "Delivery Slot (1-3)",
        1,
        3,
        2
    )

    driver_exp = st.number_input(
        "Driver Experience",
        value=5
    )

    num_stops = st.number_input(
        "Number of Stops",
        value=3
    )

# ----------------------------
# RIGHT SIDE
# ----------------------------

with col2:

    vehicle_age = st.number_input(
        "Vehicle Age",
        value=3
    )

    road_score = st.slider(
        "Road Condition Score (1-5)",
        1,
        5,
        3
    )

    package_weight = st.number_input(
        "Package Weight",
        value=15.0
    )

    fuel_eff = st.number_input(
        "Fuel Efficiency",
        value=12.0
    )

    warehouse_time = st.number_input(
        "Warehouse Processing Time",
        value=45
    )

# ----------------------------
# BUTTON
# ----------------------------

if st.button("Predict Delay Status"):

    input_data = pd.DataFrame({
        "Delivery_Distance":[delivery_distance],
        "Traffic_Congestion":[traffic],
        "Weather_Condition":[weather],
        "Delivery_Slot":[delivery_slot],
        "Driver_Experience":[driver_exp],
        "Num_Stops":[num_stops],
        "Vehicle_Age":[vehicle_age],
        "Road_Condition_Score":[road_score],
        "Package_Weight":[package_weight],
        "Fuel_Efficiency":[fuel_eff],
        "Warehouse_Processing_Time":[warehouse_time]
    })

    input_data = pd.get_dummies(input_data)

    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    prediction = model.predict(input_data)[0]

    st.header("Prediction Result")

    if prediction == 0:

        st.success(
            "✅ Predicted Status: No Delay"
        )

    else:

        st.error(
            "⚠️ Predicted Status: Delivery Delayed"
        )

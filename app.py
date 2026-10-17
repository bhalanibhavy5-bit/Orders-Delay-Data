import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
)

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Delivery Delay Dashboard",
    page_icon="🚚",
    layout="wide"
)

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/3774/3774299.png",
    width=80
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Home",
        "Dataset Overview",
        "EDA",
        "Model Training"
    ]
)

# -----------------------------
# HOME PAGE
# -----------------------------

if page == "Home":

    st.title("🚚 Delivery Delay Prediction Dashboard")

    st.markdown("""
    ### Project Objective

    This application predicts whether a delivery will be delayed using:

    - Delivery Distance
    - Traffic Congestion
    - Weather Conditions
    - Delivery Slot
    - Driver Experience
    - Number of Stops
    - Vehicle Age
    - Road Condition Score
    - Package Weight
    - Fuel Efficiency

    ### Machine Learning Models
    ✅ Logistic Regression  
    ✅ Linear SVC  
    ✅ Decision Tree  

    Developed using Streamlit & Scikit-Learn.
    """)

# -----------------------------
# FILE UPLOAD
# -----------------------------

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    TARGET = "Delivery_Delay"

    if TARGET not in df.columns:
        st.error(
            "Target Column 'Delivery_Delay' Not Found"
        )
        st.stop()

    # =========================
    # DATASET PAGE
    # =========================

    if page == "Dataset Overview":

        st.title("📊 Dataset Overview")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Rows",
            df.shape[0]
        )

        col2.metric(
            "Columns",
            df.shape[1]
        )

        col3.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

        st.subheader("Dataset Preview")

        st.dataframe(df.head())

        st.subheader("Statistical Summary")

        st.dataframe(
            df.describe()
        )

    # =========================
    # EDA PAGE
    # =========================

    elif page == "EDA":

        st.title("📈 Exploratory Data Analysis")

        st.subheader("Target Variable Distribution")

        fig, ax = plt.subplots()

        df[TARGET].value_counts().plot(
            kind="bar",
            color=["green", "red"],
            ax=ax
        )

        st.pyplot(fig)

        st.subheader("Correlation Heatmap")

        fig2, ax2 = plt.subplots(
            figsize=(10, 6)
        )

        sns.heatmap(
            df.corr(numeric_only=True),
            cmap="coolwarm",
            ax=ax2
        )

        st.pyplot(fig2)

    # =========================
    # MODEL PAGE
    # =========================

    elif page == "Model Training":

        st.title("🤖 Model Training")

        X = df.drop(columns=[TARGET])
        y = df[TARGET]

        X = pd.get_dummies(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        if st.button("Train Models"):

            results = []

            # ------------------
            # Logistic Regression
            # ------------------

            logistic_model = Pipeline([
                ("imputer",
                 SimpleImputer(strategy="median")),
                ("scaler",
                 StandardScaler()),
                ("classifier",
                 LogisticRegression(
                     max_iter=2000,
                     random_state=42
                 ))
            ])

            logistic_model.fit(
                X_train,
                y_train
            )

            y_pred_log = logistic_model.predict(
                X_test
            )

            results.append({
                "Model": "Logistic Regression",
                "Accuracy":
                accuracy_score(
                    y_test,
                    y_pred_log
                ),
                "Precision":
                precision_score(
                    y_test,
                    y_pred_log
                ),
                "Recall":
                recall_score(
                    y_test,
                    y_pred_log
                ),
                "F1":
                f1_score(
                    y_test,
                    y_pred_log
                )
            })

            # ------------------
            # Linear SVC
            # ------------------

            linear_model = Pipeline([
                ("imputer",
                 SimpleImputer(strategy="median")),
                ("scaler",
                 StandardScaler()),
                ("classifier",
                 LinearSVC(
                     random_state=42,
                     max_iter=10000
                 ))
            ])

            linear_model.fit(
                X_train,
                y_train
            )

            y_pred_svc = linear_model.predict(
                X_test
            )

            results.append({
                "Model": "Linear SVC",
                "Accuracy":
                accuracy_score(
                    y_test,
                    y_pred_svc
                ),
                "Precision":
                precision_score(
                    y_test,
                    y_pred_svc
                ),
                "Recall":
                recall_score(
                    y_test,
                    y_pred_svc
                ),
                "F1":
                f1_score(
                    y_test,
                    y_pred_svc
                )
            })

            # ------------------
       

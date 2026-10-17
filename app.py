import streamlit as st
import pandas as pd

st.title("🚚 Delivery Delay Prediction")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.subheader("Dataset Shape")

    st.write(df.shape)

    st.subheader("Columns")

    st.write(df.columns.tolist())

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    RocCurveDisplay,
    ConfusionMatrixDisplay
)

st.set_page_config(
    page_title="Delivery Delay Prediction",
    page_icon="🚚",
    layout="wide"
)

st.title("🚚 Delivery Delay Prediction Dashboard")

uploaded_file = st.file_uploader(
    "Upload Delivery Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    TARGET = "Delivery_Delay"

    if TARGET not in df.columns:
        st.error("Target column 'Delivery_Delay' not found.")
        st.stop()

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

    st.write("### Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Target Classes", y.nunique())

    if st.button("Train Models"):

        results = []

        # Logistic Regression

        logistic_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=2000,
                random_state=42
            ))
        ])

        logistic_model.fit(X_train, y_train)

        y_pred_log = logistic_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred_log)

        results.append({
            "Model": "Logistic Regression",
            "Accuracy": acc,
            "Precision": precision_score(y_test, y_pred_log),
            "Recall": recall_score(y_test, y_pred_log),
            "F1": f1_score(y_test, y_pred_log)
        })

        # Linear SVC

        linear_svc_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LinearSVC(
                max_iter=10000,
                random_state=42
            ))
        ])

        linear_svc_model.fit(X_train, y_train)

        y_pred_svc_linear = linear_svc_model.predict(X_test)

        results.append({
            "Model": "Linear SVC",
            "Accuracy": accuracy_score(y_test, y_pred_svc_linear),
            "Precision": precision_score(y_test, y_pred_svc_linear),
            "Recall": recall_score(y_test, y_pred_svc_linear),
            "F1": f1_score(y_test, y_pred_svc_linear)
        })

        # Tuned SVC

        svc_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("svc", SVC())
        ])

        param_grid = {
            "svc__kernel": ["linear", "rbf"],
            "svc__C": [0.1, 1, 10]
        }

        grid_search = GridSearchCV(
            svc_pipeline,
            param_grid,
            cv=3,
            scoring="accuracy"
        )

        grid_search.fit(X_train, y_train)

        best_svc = grid_search.best_estimator_

        y_pred_svc = best_svc.predict(X_test)

        results.append({
            "Model": "Tuned SVC",
            "Accuracy": accuracy_score(y_test, y_pred_svc),
            "Precision": precision_score(y_test, y_pred_svc),
            "Recall": recall_score(y_test, y_pred_svc),
            "F1": f1_score(y_test, y_pred_svc)
        })

        # Decision Tree

        dt_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DecisionTreeClassifier(
                random_state=42
            ))
        ])

        dt_model.fit(X_train, y_train)

        y_pred_dt = dt_model.predict(X_test)

        results.append({
            "Model": "Decision Tree",
            "Accuracy": accuracy_score(y_test, y_pred_dt),
            "Precision":

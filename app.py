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

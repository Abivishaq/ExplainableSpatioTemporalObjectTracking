import streamlit as st
import json

st.title("Multi JSON Viewer")

uploaded_files = st.file_uploader("Upload JSON files", type=["json"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.subheader(f"Contents of {uploaded_file.name}")
        try:
            json_data = json.load(uploaded_file)
            st.json(json_data, expanded=True)
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse {uploaded_file.name}: {e}")

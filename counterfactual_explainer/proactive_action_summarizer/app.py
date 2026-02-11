import os
import streamlit as st
import pandas as pd
import os
import sys

# --------- CONFIGURATION ---------
file_dir = os.path.dirname(os.path.abspath(__file__))
ce_dir = os.path.dirname(file_dir)


CSV_FOLDER = os.path.join(ce_dir,"processed_logs")
# ----------------------------------

def get_csv_files(folder):
    return [f for f in os.listdir(folder) if f.endswith(".csv")]

def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return None

def main():
    st.set_page_config(page_title="CSV Viewer", layout="wide")
    st.title("📊 CSV File Explorer")

    # Step 1: File selection
    csv_files = get_csv_files(CSV_FOLDER)
    if not csv_files:
        st.error("No CSV files found in folder.")
        return

    selected_file = st.selectbox("Select a CSV file", csv_files)
    file_path = os.path.join(CSV_FOLDER, selected_file)

    df = load_csv(file_path)
    if df is None:
        return

    st.markdown("---")
    st.subheader("🔧 Display Options")

    all_columns = df.columns.tolist()

    # Column selection
    selected_columns = st.multiselect("Choose columns to display", all_columns, default=all_columns)

    # Multi-column sort selection
    st.subheader("🔃 Sorting Options")
    sort_columns = st.multiselect("Select columns to sort by (drag to reorder)", selected_columns, default=selected_columns)
    sort_orders = {}

    for col in sort_columns:
        sort_orders[col] = st.radio(f"Sort order for '{col}'", ["Ascending", "Descending"], horizontal=True, key=f"sort_{col}")

    # Prepare sort args
    ascending_list = [sort_orders[col] == "Ascending" for col in sort_columns]

    # Filter and sort
    filtered_df = df[selected_columns]
    if sort_columns:
        filtered_df = filtered_df.sort_values(by=sort_columns, ascending=ascending_list)

    st.markdown("---")
    st.subheader("📋 Preview")
    st.dataframe(filtered_df, use_container_width=True)

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

# ---------------------------------------------------------
# 1. FRONTEND: Page Setup & Sidebar Details
# ---------------------------------------------------------
st.set_page_config(page_title="Rainfall Distribution Analyzer", layout="wide")

st.sidebar.title("Project Details")
st.sidebar.markdown("""
**Course:** Python Programming for Data Science  
**Name:** RISHI ARAVINDH L  
**Roll No:** A2544  
**Department:** 1st Year AI & DS  
**College:** K. Ramakrishna College of Engineering
""")

st.title("🌧️ Rainfall Distribution Analyzer")
st.write("Loading data automatically from the local project folder. Edit the data directly in the table to watch the analysis update.")

# ---------------------------------------------------------
# 2. DATA STORAGE & LOADING (From Local Folder)
# ---------------------------------------------------------
# Define the file name (must be in the same folder as app.py)
file_path = "tamilnadu_rainfall.csv"

# Check if the file exists in the folder before trying to load it
if os.path.exists(file_path):
    # Read the CSV directly from the folder
    df = pd.read_csv(file_path)
    
    st.subheader("Data Editor (Add, Edit, or Delete Rows)")
    st.info("💡 Click on any cell to edit it. To add a new row, scroll to the bottom. To delete a row, click the checkbox on the far left and press the Delete key.")
    
    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    
    st.sidebar.header("Chart Settings")
    
    numeric_columns = edited_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if len(numeric_columns) > 0:
        x_axis = st.sidebar.selectbox("Select X-axis (e.g., Year/Month/Region):", edited_df.columns)
        y_axis = st.sidebar.selectbox("Select Y-axis (Rainfall Amount):", numeric_columns)

        # ---------------------------------------------------------
        # 3. ENHANCED VISUALIZATIONS (Seaborn & Matplotlib)
        # ---------------------------------------------------------
        st.subheader("Live Data Visualizations")
        
        tab1, tab2, tab3 = st.tabs(["Bar Chart", "Line Chart", "Pie Chart"])
        
        with tab1:
            st.write(f"**Bar Chart: {y_axis} by {x_axis}**")
            fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
            
            sns.barplot(data=edited_df, x=x_axis, y=y_axis, ax=ax_bar, palette="mako", errorbar=None)
            
            for container in ax_bar.containers:
                ax_bar.bar_label(container, fmt='%.1f', padding=3)
            
            sns.despine()
            plt.xticks(rotation=45)
            st.pyplot(fig_bar)
            
        with tab2:
            st.write(f"**Line Chart: Trend of {y_axis} over {x_axis}**")
            fig_line, ax_line = plt.subplots(figsize=(10, 5))
            sns.lineplot(data=edited_df, x=x_axis, y=y_axis, ax=ax_line, marker="o", color="#2ca02c", linewidth=2)
            sns.despine()
            plt.xticks(rotation=45)
            st.pyplot(fig_line)
            
        with tab3:
            st.write(f"**Pie Chart: Proportion of {y_axis} by {x_axis}**")
            pie_data = edited_df.groupby(x_axis)[y_axis].sum().reset_index()
            fig_pie, ax_pie = plt.subplots(figsize=(7, 7))
            
            colors = sns.color_palette('pastel')[0:len(pie_data)]
            ax_pie.pie(pie_data[y_axis], labels=pie_data[x_axis], autopct='%1.1f%%', startangle=90, colors=colors)
            ax_pie.axis('equal') 
            st.pyplot(fig_pie)

        # ---------------------------------------------------------
        # 4. DATA ANALYSIS (Scikit-Learn)
        # ---------------------------------------------------------
        st.subheader("Machine Learning: Live Trend Prediction")
        
        if pd.api.types.is_numeric_dtype(edited_df[x_axis]):
            ml_data = edited_df.dropna(subset=[x_axis, y_axis])
            
            X = ml_data[[x_axis]].values
            y = ml_data[y_axis].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            trend_predictions = model.predict(X)
            
            fig_trend, ax_trend = plt.subplots(figsize=(10, 5))
            sns.scatterplot(data=ml_data, x=x_axis, y=y_axis, ax=ax_trend, label="Actual Edited Data", color="blue", s=100)
            ax_trend.plot(ml_data[x_axis], trend_predictions, color='red', linewidth=3, label="Scikit-Learn Trend Line")
            ax_trend.set_title(f"Linear Regression: Predicting {y_axis} based on {x_axis}")
            sns.despine()
            ax_trend.legend()
            st.pyplot(fig_trend)
        else:
            st.warning(f"To calculate the Scikit-Learn trend line, the X-axis must be numerical. Currently, '{x_axis}' is not numerical.")
            
    else:
        st.error("Please ensure your data has at least one numerical column.")

else:
    # This error shows up on the frontend if the file is missing or named incorrectly
    st.error(f"⚠️ Could not find the file `{file_path}`. Please ensure it is saved in the exact same folder as your Python script.")
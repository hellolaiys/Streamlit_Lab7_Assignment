import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Page configuration
st.set_page_config(
    page_title="Iris Flower Analysis",
    layout="wide"
)

# Set seaborn style for better looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = [iris.target_names[i] for i in iris.target]
    return df, iris

# Load data
df, iris = load_data()

# Page Title
st.title("Iris Flower Dataset Visualization")
st.markdown("Explore the famous Iris dataset using interactive filters and multiple visualization types.")

# Sidebar with filters
st.sidebar.header("Filter Options")
selected_species = st.sidebar.multiselect(
    "Select Flower Species:",
    options=df['species'].unique(),
    default=df['species'].unique()
)

# Feature selection
st.sidebar.header("Plot Settings")
x_axis = st.sidebar.selectbox(
    "X-Axis Feature:",
    options=iris.feature_names,
    index=0
)

y_axis = st.sidebar.selectbox(
    "Y-Axis Feature:",
    options=iris.feature_names,
    index=1
)

# Histogram feature
hist_feature = st.sidebar.selectbox(
    "Feature for Histogram:",
    options=iris.feature_names,
    index=2
)

# Color palette selection
palette = st.sidebar.selectbox(
    "Color Palette:",
    ["viridis", "Set2", "hus1", "pastel", "dark"]
)

# Filter data based on selections
filtered_df = df[df['species'].isin(selected_species)]

# Main Content - Visualizations
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Scatter Plot")

    fig1, ax1 = plt.subplots()
    sns.scatterplot(
        data=filtered_df,
        x=x_axis,
        y=y_axis,
        hue='species',
        palette=palette,
        s=100,
        ax=ax1
    )
    ax1.set_title(f"{y_axis} vs {x_axis}")
    ax1.set_xlabel(x_axis)
    ax1.set_ylabel(y_axis)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig1)

with col2:
    st.subheader("Distribution Histogram")

    fig2, ax2 = plt.subplots()
    for species in filtered_df['species'].unique():
        species_data = filtered_df[filtered_df['species'] == species][hist_feature]
        ax2.hist(species_data, alpha=0.6, label=species, bins=15)
    
    ax2.set_title(f"Distribution of {hist_feature}")
    ax2.set_xlabel(hist_feature)
    ax2.set_ylabel("Frequency")
    ax2.legend()
    st.pyplot(fig2)

# Data Metrics Section
st.subheader("Data Summary")

# Create columns for metrics
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric("Total Samples", len(filtered_df))

with metric_col2:
    st.metric("Number pf Species", filtered_df['species'].nunique())

with metric_col3:
    st.metric(
        f"Avg {x_axis.split(' ')[0]}",
        f"{filtered_df[x_axis].mean():.2f} cm"
    )

with metric_col4:
    st.metric(
        f"Avg {y_axis.split(' ')[0]}",
        f"{filtered_df[y_axis].mean():.2f} cm"
    )
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Iris Flower Dataset Visualization",
    layout="wide"
)

# Set seaborn style for better looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Use seaborn's built-in iris dataset
@st.cache_data
def load_data():
    df = sns.load_dataset('iris')
    df = df.rename(columns={'species': 'species'})
    return df

# Load data
df = load_data()

# App title
st.title("Iris Flower Dataset Visualization")

# Sidebar with filters
st.sidebar.header("Filter Options")

# Species filter
selected_species = st.sidebar.multiselect(
    "Select Flower Species:",
    options=df['species'].unique(),
    default=df['species'].unique()
)

# Feature selection
st.sidebar.header("Plot Settings")
feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

x_axis = st.sidebar.selectbox(
    "X-Axis Feature:",
    options=feature_columns,
    index=0
)

y_axis = st.sidebar.selectbox(
    "Y-Axis Feature:",
    options=feature_columns,
    index=1
)

# Histogram feature
hist_feature = st.sidebar.selectbox(
    "Feature for Histogram:",
    options=feature_columns,
    index=2
)

# Color palette selection
palette = st.sidebar.selectbox(
    "Color Palette:",
    ["viridis", "Set2", "husl", "pastel", "dark"]
)

# Filter data based on selections
filtered_df = df[df['species'].isin(selected_species)]

# Main content - Two visualizations only
col1, col2 = st.columns(2)

with col1:
    st.subheader("Scatter Plot")
    
    # Create scatter plot using seaborn
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
    ax1.set_xlabel(x_axis.replace('_', ' ').title())
    ax1.set_ylabel(y_axis.replace('_', ' ').title())
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig1)

with col2:
    st.subheader("Distribution Histogram")
    
    # Create histogram using matplotlib
    fig2, ax2 = plt.subplots()
    for species in filtered_df['species'].unique():
        species_data = filtered_df[filtered_df['species'] == species][hist_feature]
        ax2.hist(species_data, alpha=0.6, label=species, bins=15)
    
    ax2.set_title(f"Distribution of {hist_feature.replace('_', ' ').title()}")
    ax2.set_xlabel(hist_feature.replace('_', ' ').title())
    ax2.set_ylabel("Frequency")
    ax2.legend()
    st.pyplot(fig2)

# Data Metrics Section - Placed below visualizations
st.subheader("Data Summary")

# Create columns for metrics
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric("Total Samples", len(filtered_df))

with metric_col2:
    st.metric("Number of Species", filtered_df['species'].nunique())

with metric_col3:
    st.metric(
        f"Avg {x_axis.replace('_', ' ').title()}", 
        f"{filtered_df[x_axis].mean():.2f}"
    )

with metric_col4:
    st.metric(
        f"Avg {y_axis.replace('_', ' ').title()}", 
        f"{filtered_df[y_axis].mean():.2f}"
    )
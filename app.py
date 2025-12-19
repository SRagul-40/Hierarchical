import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier

# Set page config
st.set_page_config(page_title="Customer Segmentation App", layout="wide")

st.title("💳 Credit Card Customer Segmentation")
st.markdown("""
This app uses **Hierarchical Clustering** to group customers based on their credit usage and banking habits.
""")

# Load Data
@st.cache_data
def load_data():
    # Ensure the CSV file is in the same directory
    df = pd.read_csv("Credit Card Customer Data.csv")
    return df

try:
    df = load_data()
    
    # 1. Preprocessing (Same as your Notebook)
    X = df.drop(['Sl_No', 'Customer Key'], axis=1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Re-run Clustering (since Agglomerative doesn't have .predict())
    hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
    y_clusters = hc.fit_predict(X_scaled)
    df['Cluster'] = y_clusters
    
    # 3. Create a KNN Classifier to "predict" new inputs
    # This allows us to assign a new user input to the nearest cluster
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_scaled, y_clusters)

    # Sidebar for User Input
    st.sidebar.header("User Input Features")
    def user_input_features():
        avg_limit = st.sidebar.number_input("Avg Credit Limit", min_value=3000, max_value=200000, value=50000)
        total_cards = st.sidebar.slider("Total Credit Cards", 1, 10, 4)
        visits_bank = st.sidebar.slider("Total Visits Bank", 0, 5, 2)
        visits_online = st.sidebar.slider("Total Visits Online", 0, 15, 1)
        calls_made = st.sidebar.slider("Total Calls Made", 0, 10, 3)
        
        data = {
            'Avg_Credit_Limit': avg_limit,
            'Total_Credit_Cards': total_cards,
            'Total_visits_bank': visits_bank,
            'Total_visits_online': visits_online,
            'Total_calls_made': calls_made
        }
        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()

    # Display App Layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("User Input")
        st.write(input_df)
        
        if st.button("Predict Cluster"):
            # Scale the input
            input_scaled = scaler.transform(input_df)
            prediction = knn.predict(input_scaled)
            
            cluster_id = prediction[0]
            st.success(f"Target Customer belongs to: **Cluster {cluster_id}**")
            
            # Explain the Cluster based on your notebook means
            if cluster_id == 0:
                st.info("💡 **Cluster 0:** Low credit limit, high bank visits, many phone calls.")
            elif cluster_id == 1:
                st.info("💡 **Cluster 1:** High credit limit, low bank visits, high online visits.")
            else:
                st.info("💡 **Cluster 2:** Medium credit limit, moderate card usage.")

    with col2:
        st.subheader("Cluster Visualization")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            data=df, 
            x='Avg_Credit_Limit', 
            y='Total_visits_online', 
            hue='Cluster', 
            palette='viridis', 
            s=100, 
            ax=ax
        )
        # Plot the user's input point
        st.pyplot(fig)

    # Show Dataset
    if st.checkbox("Show Raw Data with Clusters"):
        st.write(df.head(20))

except FileNotFoundError:
    st.error("Error: 'Credit Card Customer Data.csv' not found. Please upload the dataset to the app folder.")

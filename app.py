import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import ast


# Load and prepare data
df = pd.read_csv("Worldwide Travel Cities Dataset (Ratings and Climate).csv")

df['avg_temp_monthly'] = df['avg_temp_monthly'].apply(ast.literal_eval)

st.title("Travel Destination Recommender")
st.write("Find your next travel destination based on your preferences!")

rating_columns = ['culture', 'adventure', 'nature', 'beaches', 'nightlife', 'cuisine', 'wellness', 'urban']

# Get user preferences with sliders
st.subheader("Rate your preferences (0-5)")
user_input = {}
for col in rating_columns:
    user_input[col] = st.slider(col.capitalize(), 0.0, 5.0, 2.5)

# Scale and recommend
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[rating_columns])

user_vec = np.array([user_input[col] for col in rating_columns]).reshape(1, -1)
user_vec_scaled = scaler.transform(user_vec)

similarities = cosine_similarity(user_vec_scaled, X_scaled)[0]
df['Similarity'] = similarities

top_n = 10
top_matches = df.sort_values(by='Similarity', ascending=False).head(top_n)

st.subheader(f"Top {top_n} Recommendations")
st.dataframe(top_matches[['city', 'country', 'Similarity'] + rating_columns])
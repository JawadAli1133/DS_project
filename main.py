import streamlit as st
import joblib
import pandas as pd
import random

# Load the trained SVD++ model
svd_pp = joblib.load('svd_model.pkl')

# Load the movies dataframe
movies = pd.read_csv('movies.csv')

# Load the top_n dictionary (user-wise movie recommendations)
top_n = joblib.load('top_n.pkl')


# Function to get recommended movies for a user
def Generate_Recommended_Movies(u_id, n=10):
    if u_id not in top_n:
        return pd.DataFrame()  # Return an empty dataframe if no recommendations for this user
    
    recommend = pd.DataFrame(top_n[u_id], columns=["Movie_Id", "Predicted_Rating"])
    recommend = recommend.merge(movies, how="inner", left_on="Movie_Id", right_on="movieId")
    recommend = recommend[["Movie_Id", "title", "genres", "Predicted_Rating"]]
    
    return recommend[:n]

# Streamlit UI
st.title('Movie Recommendation System')

# User input for User ID
user_id = st.number_input('Enter your User ID:', min_value=1, max_value=100000)  # Adjust range based on your data

# When the button is clicked, generate movie recommendations
if st.button('Recommend Movie'):
    if user_id in top_n:
        recommended_movies = Generate_Recommended_Movies(user_id)
        st.write(f"Top recommended movies for User {user_id}:")
        st.write(recommended_movies)
    else:
        st.write("Sorry, we don't have recommendations for this user ID.")

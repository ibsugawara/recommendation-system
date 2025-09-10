import streamlit as st
import requests
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_animes():
    return pd.read_csv("artifacts/anime_list.csv")

@st.cache_data
def load_embeddings():
    return np.load("artifacts/anime_embeddings.npy")

animes = load_animes()
embeddings = load_embeddings()

def fetch_image(mal_id):
    url = f"https://api.jikan.moe/v4/anime/{mal_id}"
    data = requests.get(url).json()
    image_url = data['data']['images']['jpg']['image_url']

    return image_url
    
def recommend(anime_name, top_k=5):
    idx = animes[animes['title'] == anime_name].index[0]
    anime_vec = embeddings[idx]
    sims = cosine_similarity([anime_vec], embeddings)[0]
    
    top_indices = sims.argsort()[-top_k-1:][::-1]
    recommended_name = []
    recommended_image = []
    
    for i in top_indices:
        if i != idx:
            recommended_name.append(animes.iloc[i]['title'])
            recommended_image.append(fetch_image(animes.iloc[i]['mal_id']))
        if len(recommended_name) == top_k:
            break
    return recommended_name, recommended_image

st.header("Anime Recommendation System")

anime_list = animes['title'].values
selected_anime = st.selectbox('Choose an anime to get a recommendation',
             anime_list)

if st.button('Show recommendation'):
    recommended_anime_name, recommended_anime_image = recommend(selected_anime)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(recommended_anime_image[0], width=150)
        st.text(recommended_anime_name[0])


    with col2:
        st.image(recommended_anime_image[1], width=150)
        st.text(recommended_anime_name[1])

    
    with col3:
        st.image(recommended_anime_image[2], width=150)
        st.text(recommended_anime_name[2])


    with col4:
        st.image(recommended_anime_image[3], width=150)
        st.text(recommended_anime_name[3])


    with col5:
        st.image(recommended_anime_image[4], width=150)
        st.text(recommended_anime_name[4])

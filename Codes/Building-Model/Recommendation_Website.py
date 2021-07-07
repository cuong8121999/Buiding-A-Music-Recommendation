#!/usr/bin/env python
# coding: utf-8

# %%
# Import Necessary Packages for Building a Web App
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
import scipy.sparse as sparse
from numpy import loadtxt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st
import implicit

import warnings
warnings.filterwarnings('ignore')

# %%
# Create a general path
general_path = '/Users/user/Documents/Đồ án tốt nghiệp/Code'
general_path_audio = '/Users/user/Documents/Đồ án tốt nghiệp/Music Data/Data/genres_original/combine_audio'
# Import data for CF and Content-based
cf_df = pd.read_csv(f'{general_path}/reduced_cf_df.csv')
cb_df = pd.read_csv(f'{general_path}/Combined_data_audio.csv')

# %%
bic_score = pd.read_csv(f'{general_path}/bic_score.csv')

# %%
# Import data for choosing K
with open(f'{general_path}/sum_distance.pkl', 'rb') as pickle_in:
    sum_distances = pickle.load(pickle_in)

with open(f'{general_path}/silhouette.pkl', 'rb') as pickle_in:
    silhouette = pickle.load(pickle_in)

# %%
# Create dataframe for sum_distance, silhouette and BIC
sum_distances_df = pd.DataFrame()
sum_distances_df['inertia_score'] = sum_distances
sum_distances_df['k_num'] = list(range(1,25))

silhouette_df = pd.DataFrame()
silhouette_df['silhouette_score'] = silhouette
silhouette_df['k_num'] = list(range(2,25))

bic_score_df = pd.DataFrame()


# %%
# Standardize data before training
cols = cb_df.columns[1:-1]
scale = StandardScaler()
scaled_cb_df = scale.fit_transform(cb_df.iloc[:,1:-1])

# Create a new scaled dataframe for content-based
scaled_cb_df = pd.DataFrame(scaled_cb_df)
scaled_cb_df.columns = cols
scaled_cb_df = pd.concat([cb_df[['filename']], scaled_cb_df, cb_df[['label']]], axis=1)

# %%
# Reduce dimesions of dataset for visualization
reduce_dim = TSNE(2, random_state=1)
two_dim = reduce_dim.fit_transform(scaled_cb_df.iloc[:,1:-1])

# Create a new reduced dimension dataframe
reduced_cb = pd.DataFrame(two_dim, columns=['X','Y'])
reduced_cb_df = pd.concat([cb_df[['filename']], reduced_cb, cb_df[['label']]], axis=1)

# %%
# Calculate the cosine similarity between songs
similar = scaled_cb_df.copy()
similar = similar.set_index('filename')
similar_df = similar.iloc[:,:-1]
# Cosine Similarity
similarity = cosine_similarity(similar_df)

# Convert into a dataframe
sim_df_labels = pd.DataFrame(similarity)
sim_df = sim_df_labels.set_index(similar.index)
sim_df.columns = similar.index

# %%
# The implicit library expects data as item-user matrix
sparse_item_user = sparse.csr_matrix((cf_df['times'].astype(float), (cf_df['song_artist_cat'], cf_df['user_id_cat'])))
sparse_user_item = sparse.csr_matrix((cf_df['times'].astype(float), (cf_df['user_id_cat'], cf_df['song_artist_cat'])))

# Calculate sparsity of matrix
matrix_size = sparse_item_user.shape[0]*sparse_item_user.shape[1]
num_purchases = len(sparse_item_user.nonzero()[0])
sparsity = round(100*(1 - (num_purchases/matrix_size)), 2)
# %%

# Image
image = st.image(f"{general_path}/music_background.jpg")

# Set the title for Web
title = st.title("Music Recommendation System")

# Create a selectbox for selecting method
method_recommend = st.sidebar.selectbox(
    "Select A Style", ("Explore Your Neighbours?", "Explore Your World?"))

# Depending the method to choose the algorithm
if method_recommend == "Explore Your Neighbours?":

    # Present data of content-based
    header_data = st.header("Data Information:")
    table = st.dataframe(scaled_cb_df.sample(20, random_state=1))

    # Visualize by dot plot
    header_vis = st.header("Data Visualization:")
    chart = st.vega_lite_chart(reduced_cb_df, {
        'height': 400,
        'width': 700,
        'mark': 'point',
        'encoding': {
            'x': {'field': 'X', 'type': 'quantitative'},
            'y': {'field': 'Y', 'type': 'quantitative'},
            'color': {'field': 'label', 'type':'nominal'},
            'shape': {'field': 'label', 'type': 'nominal'}
        }
    })

    # Select Algorithm
    algorithm = st.sidebar.selectbox("Select a Algorithm", 
    ("K-Means",
    "Gaussian Mixture Model"))
    header_al = st.header(f'**{algorithm}**')

    # Describe the steps in model
    describe = st.write("""
    ### Steps for choosing the best K for Clustering model
    1. Set a range of K for model
    2. Choose the best optimal number of K for model
    3. Visualize the clustered data after using model
    4. Run model and return the recommended song
    """)

    if algorithm == "K-Means":
    # Choose the best K
        header_k = st.subheader("Graphs for Choosing K clusters:")
        nertia = st.write("**_Inertia Score_**")
        inertia_chart = st.vega_lite_chart(sum_distances_df, {
                            'height': 300,
                            'width': 650,
                            'mark': {
                                'type': 'line',
                                'point': True
                            },
                            'encoding': {
                                'x': {'field': 'k_num', 'type': 'quantitative'},
                                'y': {'field': 'inertia_score', 'type': 'quantitative'}
                            }
                        })

        silhouette = st.write("**_Silhouette Score_**")
        silhouette_chart = st.vega_lite_chart(silhouette_df, {
                            'height': 300,
                            'width': 650,
                            'mark': {
                                'type': 'line',
                                'point': True
                            },
                            'encoding': {
                                'x': {'field': 'k_num', 'type': 'quantitative'},
                                'y': {'field': 'silhouette_score', 'type': 'quantitative'}
                            }       
                        })
    else:
        header_k = st.subheader("Graphs for Choosing K clusters:")
        bic = st.write("**_BIC Score_**")
        fig = px.bar(bic_score, x="k", y=["spherical", "tied", "diag", "full"], barmode='group', height=500)
        # st.dataframe(df) # if need to display dataframe
        bic_chart = st.plotly_chart(fig)

    # Visualize clustered data After using models
    header_cluster = st.subheader(f"Graph for Clustered Data by {algorithm}:")
    k_num = st.sidebar.slider("Select the number of group:", 1, 25)

    # Depend on specified model is to train model
    def get_k(k, al):
        reduced_model = reduced_cb.copy()
        if al == "K-Means":
            model = KMeans(n_clusters=k, n_init=10, random_state=1)
        else:
            model = GaussianMixture(n_components=k, n_init=10, random_state=1)

        model.fit(scaled_cb_df.iloc[:,1:-1])    
        result_model = model.predict(scaled_cb_df.iloc[:,1:-1])
        reduced_model['cluster'] = result_model
        return reduced_model

    # Activate the get_k function
    model_k = get_k(k_num, algorithm)

    # Visualize by dot plot for clustered data
    clustered = st.vega_lite_chart(model_k, {
            'height': 400,
            'width': 700,
            'mark': 'point',
            'encoding': {
                'x': {'field': 'X', 'type': 'quantitative'},
                'y': {'field': 'Y', 'type': 'quantitative'},
                'color': {'field': 'cluster', 'type':'nominal'},
                'shape': {'field': 'cluster', 'type': 'nominal'}
                }
            })

    # Function for recommendation 
    name_func = st.header("Song Recommendation:")
    song_name = st.sidebar.selectbox("Select a Song:", list(sim_df.index))
    heading_song = st.subheader(f"List of recommended song by {song_name}")

    # Depend on specified model is to train model
    def recommend_song(song):
        sim_df['cluster'] = list(model_k.iloc[:,-1])
        # Take the cluster
        cluster = sim_df['cluster'][song]
        # Filter songs are same genre
        series = sim_df[sim_df['cluster'] == cluster]
        #Sort the value and fill out the highest scores
        shortest_distance = series[song].sort_values(ascending=False)
        return shortest_distance.head(11)

    # Activate the function
    recommend_cb = recommend_song(song_name)
    recommended_df = st.table(recommend_cb)

    # Function for filtering audio
    heading_audio = st.write("Audio for each song:")
    def audio_song(df_song):
        list_song = list(recommend_cb.index)
        specify = st.write(f"Specified song ({list_song[0]})")
        audio_specify = st.audio(f'{general_path_audio}/{list_song[0]}', format="mp3/wav")
        for i in list_song[1:]:
            audio_tag = st.write(f"{i}")
            audio_file = st.audio(f'{general_path_audio}/{i}', format="mp3/wav")
    
    audio = (audio_song(recommend_cb))

else:
    header_al = st.header('Alternative Least Square')

    # Present data of content-based
    col1, col2 = st.beta_columns([4,1])
    with col1:
        header_data = st.subheader("Data Information:")
        table = st.table(cf_df.iloc[:,:-2].sample(5, random_state=1))
    with col2:
        header_describe = st.subheader("Sparsity:")
        describe = st.write(f"*{sparsity}*%")

    # header for pick up a user
    for_user = st.header("Pick up a user and I'll show u:")

    # User and Item selection
    user_id = st.sidebar.selectbox("User:", list(np.unique(cf_df.iloc[:,-2])))
    item_id = st.sidebar.selectbox("Item:", list(np.unique(cf_df.iloc[:,1])))

    # Seed for random_state
    seed = st.sidebar.selectbox("Random State:", ("Stable", "Constantly Changing"))


    # Initialize the als model and fit it using the sparse item_user matrix
    # Custom parameter for model
    factor_val = st.sidebar.select_slider("Factors", list(range(20,101,5)))
    regularization_val = st.sidebar.select_slider("Regularization", [0.0001, 0.001, 0.01, 0.05, 0.1])
    iteration_val = st.sidebar.select_slider("Iteration", list(range(15,51,5)))

    # Calculate the cofidence by multiplying it by our alpha value
    alpha_val = st.sidebar.select_slider("Alpha", list(range(15,51,5)))
    data_conf = (sparse_item_user * alpha_val).astype('double')

    if seed == "Stable":
        model_cf = implicit.als.AlternatingLeastSquares(factors=factor_val, regularization=regularization_val, iterations=iteration_val, use_cg=True, random_state=1)
        model_cf.fit(data_conf)
    else:
        model_cf = implicit.als.AlternatingLeastSquares(factors=factor_val, regularization=regularization_val, iterations=iteration_val, use_cg=True)
        model_cf.fit(data_conf)

    header_user = st.subheader(f":musical_keyboard:. 20 songs are listened the most by user {user_id}:")
    songs_pop = st.table(cf_df[cf_df['user_id_cat'] == user_id].sort_values("times", ascending=False).head(20).iloc[:,1:3])

    # Function for recommending songs for user
    def recommend_song_cf(user):
        # The most songs are listened by user
        recommended = model_cf.recommend(user, sparse_user_item, N=10, filter_already_liked_items=True)
    
        song = []
        scores = []
    
        # Get artist names from ids
        for item in recommended:
            idx, score = item
            song.append(cf_df['song_artist'].loc[cf_df['song_artist_cat'] == idx].iloc[0])
            scores.append(score)
        
        recommendations = pd.DataFrame({'song_artist': song, 'score': scores})
        return recommendations

    header_user_song = st.subheader(":musical_keyboard::musical_keyboard:. 10 songs are recommended for him/her: ")     
    user_song = st.table(recommend_song_cf(user_id))

    # Function for recommending similar songs with specified song
    def similar_song(item):
        item_cat = cf_df['song_artist_cat'][cf_df['song_artist'] == item_id].iloc[0]
        similar = model_cf.similar_items(item_cat, 11)
    
        song = []
        scores = []
        for item in similar:
            idx, score = item
            song.append(cf_df['song_artist'].loc[cf_df['song_artist_cat'] == idx].iloc[0])
            scores.append(score)
        
        similar_songs = pd.DataFrame({'song_artist': song, 'scores': scores})
        return similar_songs
    
    for_item = st.header("Or Pick up a song and I'll show u: ")
    header_song_song = st.subheader(f":cd: 10 Similar songs with '**_{item_id}_**' song:")
    song_song = st.table(similar_song(item_id))
    

# %%

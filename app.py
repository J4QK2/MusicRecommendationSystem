import streamlit as st
import requests
import pandas as pd
import joblib as jb
from sklearn.metrics.pairwise import cosine_similarity

recData = jb.load("recommendationData.pkl")
simData = jb.load("E:/big models/similarities.pkl")

recDataFrame = pd.DataFrame(recData)


def recommend(name):
    music_row = recDataFrame[recDataFrame['name'].str.contains(name, case=False, na=False, regex=False)]

    if music_row.empty:
        st.write("OOps no song in the dataset")
    
    music_index = music_row.index[0]
    music_vector = simData[music_index]
    music_vector = music_vector.reshape(1, -1)

    similarities = cosine_similarity(music_vector, simData)
    distances = similarities.flatten()
    
    music_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommendList = []

    for i in music_list:
        music_name = recDataFrame.iloc[i[0]]['name']
        recommendList.append(music_name)
    
    return recommendList

selected_music_name = st.selectbox("Select a music you like", recDataFrame["name"].values)



if st.button("Recommend"):
    names = recommend(selected_music_name)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(names[0])
    with col2:
        st.text(names[1])
    with col3:
        st.text(names[2])
    with col4:
        st.text(names[3])
    with col5:
        st.text(names[4])
    
    
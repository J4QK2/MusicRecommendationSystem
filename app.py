import streamlit as st
import requests
import pandas as pd
import joblib as jb

recData = jb.load("recommendationData.pkl")
simData = jb.load("similarities.pkl")

recDataFrame = pd.DataFrame(recData)

def get_poster(name):
    response = requests.get("".format(name))

    data = response.json()
    
    return data

def recommend(name):
    music_row = recDataFrame[recDataFrame['name'].str.contains(name, case=False, na=False)]

    if music_row.empty:
        st.write("OOps no song in the dataset")
    
    music_index = music_row.index[0]
    distances = simData[music_index]

    music_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommendList = []
    recommendPoster = []

    for i in music_index:
        music_name = music_list.iloc[i[0]]['name']
        recommendList.append(music_name)
        recommendPoster.append(get_poster(music_name))
    
    return recommendList, recommendPoster


    
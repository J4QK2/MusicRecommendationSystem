import streamlit as st
import requests
import pandas as pd
import joblib as jb
from sklearn.metrics.pairwise import cosine_similarity

recData = jb.load("recommendationData.pkl")
simData = jb.load("E:/big models/similarities.pkl")

recDataFrame = pd.DataFrame(recData)

st.set_page_config(layout="wide")
st.title("Recommendation System")

def get_poster(name):
    url = f"https://api.deezer.com/search?q={name}"

    try:
        res = requests.get(url)
        data = res.json()

        if 'data' in data and len(data['data']) > 0:
            return data['data'][0]['album']['cover_big']
    
    except Exception as e:
        print("api error: ", e)
        return 0

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
    recommendPosterList = []

    for i in music_list:
        music_name = recDataFrame.iloc[i[0]]['name']
        recommendList.append(music_name)
        recommendPosterList.append(get_poster(recDataFrame.iloc[i[0]]['name']))

    
    return recommendList, recommendPosterList

selected_music_name = st.selectbox("Select a music you like", recDataFrame["name"].values)



if st.button("Recommend"):
    names, posters = recommend(selected_music_name)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(names[0])
        if posters[0]:
            st.image(posters[0])
        else:
            st.write("There is no poster for this music")
    with col2:
        st.text(names[1])
        if posters[1]:
            st.image(posters[1])
        else:
            st.write("There is no poster for this music")
    with col3:
        st.text(names[2])
        if posters[2]:
            st.image(posters[2])
        else:
            st.write("There is no poster for this music")
    with col4:
        st.text(names[3])
        if posters[3]:
            st.image(posters[3])
        else:
            st.write("There is no poster for this music")
    with col5:
        st.text(names[4])
        if posters[4]:
            st.image(posters[4])
        else:
            st.write("There is no poster for this music")
    
    
import collections
from itertools import combinations
import streamlit as st
import pandas as pd
from PIL import Image
import publi_recommender
from os.path  import exists
import os
import re


def check_password():
    """Returns `True` if the user had the correct password."""
    return True
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

# def add_classical_genre(df_movies):
#     df_movies['Year'] = df_movies['title'].apply(lambda x: get_year(x))
#     for index, row in df_movies.iterrows():
#         year = row['Year']
#         genres = row['genres']

#         if int(year) < 2000:
#             genres = genres + '|classic'

#         df_movies.at[index, 'genres'] =  genres
    
#     df_movies = df_movies.drop('Year', axis=1)
#     return df_movies

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.sidebar.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

if __name__=='__main__':

    if check_password():

        # st.write("""
        # # Demo Recommender System
        # """)
        st.set_page_config(  # Alternate names: setup_page, page, layout
        layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
        initial_sidebar_state="expanded",  # Can be "auto", "expanded", "collapsed"
        page_title='AI Suite: Publi Recommender',  # String or None. Strings get appended with "• Streamlit". 
        page_icon='demo_images/fav_icon_1.png',  # String, anything supported by st.image, or None.
        )
        st.write('<style>body { margin: 0; font-family: Arial, Helvetica, sans-serif;} \
            .sticky { position: fixed; top: 0; width: 100%;} \
            </style><div class="header" id="myHeader"> <font size="+5"> <b>'+str('AI Suite > Publi Recommendation System')+'</b> </font></div>', \
            unsafe_allow_html=True)
            
        st.sidebar.image('demo_images/logo.png')

        uploaded_file = st.sidebar.file_uploader("Choose a video...")
        if uploaded_file is not None:
            vid = uploaded_file.name
            st.sidebar.success('Video uploaded!')

            print(vid)

            if( os.path.splitext(vid)[0] == '6863'):
                print('ok')
            
                title = st.sidebar.text_input('Movie title', 'Men in Black')
                st.sidebar.write('The current movie title is: ', title)

                description = st.sidebar.text_input('Movie description', 'After a police chase with an otherworldly being, a New York City cop is recruited as an agent in a top-secret organization established to monitor and police alien activity on Earth: the Men in Black. Agent Kay and new recruit Agent Jay find themselves in the middle of a deadly plot by an intergalactic terrorist who has arrived on Earth to assassinate two ambassadors from opposing galaxies.')
                st.sidebar.write('The current movie description is: ', description)

                bytes_data = uploaded_file.getvalue()
                st.video(bytes_data, format="video/mp4", start_time=0)

                if st.button('Generate Ads'):
                    with st.spinner('calculating'):
                        df_similarity, df_ads = publi_recommender.excect('6863.csv', title, description)

                    # video_file = open('videos\\1580.mp4', 'rb')
                    # video_bytes = video_file.read()
                    # st.video(video_bytes)
                        
                    ##Best match
                    top = df_similarity["Sum"].nlargest(10).index.tolist()
                    for i in top:
                        st.write("Ad recommender with similarity: ", df_similarity.iloc[i])
                        st.write("Word best similarity is: ", df_similarity.iloc[i, 0:len(df_similarity.columns) - 1].idxmax(), df_similarity.iloc[i, 0:len(df_similarity.columns) - 1].max())
                        st.write(df_ads.iloc[i,1:4])
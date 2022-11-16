import collections
from itertools import combinations
import streamlit as st
from os import listdir
import torch
import time
from os.path import isfile, join
import pandas as pd
from PIL import Image
import publi_recommender
import video_tags
from os.path  import exists
import os
import re

execution_path = os.getcwd()

def detect(video_path):

    model_y = torch.hub.load('ultralytics/yolov5', 'yolov5l')  # or yolov5n - yolov5x6, custom

    df_video = pd.DataFrame()
    head, tail = os.path.split("videos/" + video_path)
    filename = os.path.splitext(tail)[0]
    scene_list = video_tags.find_scenes("videos/" + video_path) #scene_list is now a list of FrameTimecode pairs representing the start/end of each scene
    video_tags.split_video_ffmpeg("videos/" + video_path, scene_list) #en mypath están ahora los vídeos de cada escena

    scenes = [f for f in listdir(execution_path) if f.startswith(filename)]
    progress = 1/len(scenes)
    my_bar = st.progress(0)
    numbers = st.empty()
    for i, scene in enumerate(scenes):
        df_scene=pd.DataFrame()
        pathIn = join(execution_path, scene)
        pathOut = join(execution_path,"scenes\\")
        video_tags.video2frames(pathIn, pathOut)

        frames = [f for f in listdir(pathOut) if isfile(join(pathOut, f))]
        for j, frame in enumerate(frames):            
            img = join(pathOut, frame)  # or file, Path, PIL, OpenCV, numpy, list
            df_places = video_tags.places_prediction(img)
            results = model_y(img)
            df_frame = results.pandas().xyxy[0]  # or .show(), .save(), .crop(), .pandas(), etc.
            df_frame = df_frame.append(df_places)
            df_frame.insert(0, 'frame', j)
            df_scene = df_scene.append(df_frame)

        df_scene.insert(0, 'scene', i)
        df_video = df_video.append(df_scene, ignore_index=True)
        video_tags.delete_files_in(pathOut)
        my_bar.progress((i+1)*progress)
        with numbers.container():
            st.write(round((i+1)*progress*100,2), "%")
    
    df_video.to_csv(filename + ".csv")
    st.write("Finished!!! To continue click on Generate Ads")
    st.balloons()
    # return df_video

def check_password():
    """Returns `True` if the user had the correct password."""
    # return True
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


if __name__=='__main__':

    if check_password():

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

        st.sidebar.checkbox("Using a pre-computed video", key="disabled")
        precalculated_videos_df = pd.read_csv('precalculated_videos.csv', header = 0, skipinitialspace = True, encoding = "ISO-8859-1")
        option = st.sidebar.selectbox(
        'Please select one of the pre-calculated .CSV',
        precalculated_videos_df,
        disabled = not st.session_state.disabled)
        uploaded_file = st.sidebar.file_uploader("Choose a video...", disabled=st.session_state.disabled)

        video_path = ""
        if not st.session_state.disabled:

            title = st.sidebar.text_input('Movie title', "Title...")
            description = st.sidebar.text_input('Movie description', "Description...")

            if uploaded_file is not None:
                video_path = uploaded_file.name
                st.sidebar.success('Video uploaded!')
                bytes_data = uploaded_file.getvalue()
                st.video(bytes_data, format="video/mp4", start_time=0)
                
                if st.button('Calculate video tags'):
                    with st.spinner('calculating video tags...'):
                        # video_tags.excect("videos/" + video_path)
                        detect(video_path)
                        
        else:
            video_path = precalculated_videos_df.loc[precalculated_videos_df['title'] == option, 'csv'].iloc[0]
            
            title_t = option
            description_t = precalculated_videos_df.loc[precalculated_videos_df['title'] == option, "description"].iloc[0]
            
            title = st.sidebar.text_input('Movie title', title_t)
            description = st.sidebar.text_input('Movie description', description_t)


        if(video_path != "" and video_path != "nan"):
            print('ok')                    

            video_tags = os.path.splitext(video_path)[0] + ".csv"
            if st.button('Generate Ads'):
                with st.spinner('calculating similar ads...'):
                    df_similarity, df_ads = publi_recommender.excect(video_tags, title, description)

                top = df_similarity["Sum"].nlargest(10).index.tolist()
                for i in top:
                    st.write("Ad recommender with similarity: ", df_similarity.iloc[i])
                    st.write("Word best similarity is: ", df_similarity.iloc[i, 0:len(df_similarity.columns) - 1].idxmax(), df_similarity.iloc[i, 0:len(df_similarity.columns) - 1].max())
                    st.write(df_ads.iloc[i,1:4])

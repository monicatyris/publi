#importing libraries
import numpy as np
import pandas as pd
import seaborn  as sns
import re
import matplotlib.pyplot as plt
from keybert import KeyBERT
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import gensim.downloader
from sklearn import preprocessing
import spacy

def get_ads():
    #getting dataset with details
    cols = ['Year','Product Type','Product/Title','Plot/Notes', 'Tags']
    df_ads = pd.read_csv('superbowl-ads-tags2.csv',header = None,names = cols,encoding = "ISO-8859-1")
    df_ads = df_ads.drop(0)
    df_ads['Plot/Notes'] = df_ads['Plot/Notes'].fillna("")
    df_ads = df_ads.astype({"Year": int, "Product Type": str, "Product/Title": str, "Plot/Notes": str})

    return df_ads

def get_tags(description, df_video, kw_model):
    # nlp = spacy.load("en_core_web_sm")  
    kws = kw_model.extract_keywords(description, keyphrase_ngram_range=(1, 1), stop_words='english')
    # sents = nlp(kws) 
    # print([ee for ee in kws.ents if ee.label_ == 'PERSON'])
    # print("Keywords Note: ", kw, type(kw))
    for kw in kws:
        new_row = {'name': kw[0], 'confidence': kw[1]}
        df_video = df_video.append(new_row, ignore_index=True)

    return df_video

def extract_top_keywords_video(title, description, df_video, topn=5):    

    kw_model = KeyBERT(model='all-mpnet-base-v2')

    df_video = get_tags(description, df_video, kw_model)
    # print(df_video)
    df_video = get_tags(title, df_video, kw_model)
    # print(df_video)

    # df_video = get_genres_tags(video_ID, df_video_descr, df_video)
    df_video = df_video.sort_values('confidence', ascending=False)
    df_video = df_video.drop_duplicates(subset=['name'], keep='first')
    return df_video

def get_similars(df_video, glove_vectors, topn = 5):

    for index, row in df_video.iterrows():
        word = row['name'].split("/")[0].replace('_', '-').replace(' ', '-')
        if(word in glove_vectors.key_to_index):
            similar_words = [x[0] for x in glove_vectors.most_similar(word, topn=topn)]
            confidence = [row['confidence']]*len(similar_words)
            df = pd.DataFrame({'name':similar_words,'confidence':confidence})
            df_video = pd.concat([df_video, df])

    df_video = df_video.sort_values('confidence', ascending=False)
    return df_video

def get_video_info(path, title, description):
    #getting dataset with details
    cols = ['scene','confidence','class', 'name']
    df_video = pd.read_csv(path, header = 0,skipinitialspace=True,usecols=cols,encoding = "ISO-8859-1")
    df_video = df_video[(df_video.name != "person")]
    df_video = df_video[(df_video.name != "natural light")]
    # df_video = df_video[(df_video.scene <= 10)]
    df_video = df_video.loc[df_video['confidence'] >= 0.5,:].groupby(['name'], as_index=False)['confidence'].sum()
    # print(df_video)

    # df_video['confidence']=(df_video['confidence']-df_video['confidence'].mean())/df_video['confidence'].std()
    df_video['confidence']=(df_video['confidence']-df_video['confidence'].min())/(df_video['confidence'].max()-df_video['confidence'].min())

    cols = ['movieId','title','genres','description','keywords']
    # df_video_descr = pd.read_csv('C:\\Users\\Monica\\Documents\\Tyris\\Publi\\Notebooks\\youtube_trailers\\movies_plus.csv',header = 0,skipinitialspace=True,usecols=cols,encoding = "ISO-8859-1")

    df_video = extract_top_keywords_video(title, description, df_video)
    # df_video = get_similars(df_video)

    # df_video.head(20)
    df_video = df_video.astype({"confidence": float, "name": str})

    return df_video

def get_similarity(video_tag, video_tag_conf, ad_tags, model):
    ad_tags = list(eval(ad_tags))
    similarity = 0
    for tag in ad_tags:
        ad_tag = tag[0].replace(" ", "-")
        try:
            s = model.similarity(video_tag, ad_tag)
            #*tag[1]*video_tag_conf
        except:
            s = 0
        print("similarity: ", video_tag, ad_tag, "===", s )
        similarity+=s
    return similarity


def get_similarity_matrix(df_video, ads_tags, model, top_n=10):
    video_tags = list(df_video['name'].head(top_n))
    video_tags_conf = list(df_video['confidence'].head(top_n))

    # total_similarity = 0
    df_similarity = pd.DataFrame()
    # for idxc, video_tag in enumerate(video_tags):
    for idxr, ad_tags in enumerate(ads_tags):
        print("*****     Ad : ", idxr, "     *****")
        for idxc, video_tag in enumerate(video_tags):
            video_tag = video_tag.replace(' ', '-')
            video_tag = video_tag.replace('_', '-')
            video_tag = video_tag.split("/")[0]
            # print(video_tag, video_tags_conf[idxr], idxr, idxc)
            df_similarity.at[idxr, idxc] = get_similarity(video_tag, video_tags_conf[idxc], ad_tags, model)
    return df_similarity



def excect(video, title, description, precalculated = True):
    df_ads = get_ads()
    
    if(precalculated):
        pass
        precalculated_videos_df = pd.read_csv('precalculated_videos.csv', header = 0, skipinitialspace = True, encoding = "ISO-8859-1")
        sm_path = precalculated_videos_df.loc[precalculated_videos_df['title'] == title, "similarity_matrix"].iloc[0]
        df_similarity = pd.read_csv(sm_path, header = 0, skipinitialspace = True, encoding = "ISO-8859-1")

    else:
        glove_vectors = gensim.downloader.load('fasttext-wiki-news-subwords-300')
        df_video = get_video_info(video, title, description)

        df_similarity = get_similarity_matrix(df_video, df_ads["Tags"], glove_vectors, top_n=5)
        df_similarity = pd.DataFrame(df_similarity.values, columns = list(df_video['name'].head(5)))
        df_similarity["Sum"] = df_similarity.sum(axis=1)
        df_similarity.to_csv("similarity_matrix.csv",index=False)

    return df_similarity, df_ads
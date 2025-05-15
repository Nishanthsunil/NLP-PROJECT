import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import requests

le=LabelEncoder()
feature=TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)



# spam detection datapreparation
spam_detection_model=pickle.load(open("C:\\Users\\Nishanth S\\Desktop\\class\\NLP_project\\spam_detection.sav","rb"))
data_spam=pd.read_csv("C:\\Users\\Nishanth S\\Desktop\\class\\NLP_project\\smsspamcollection.tsv",sep='\t')
data_spam['label']=le.fit_transform(data_spam['label'])
x=data_spam['message']
y=data_spam['label']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
x_train=feature.fit_transform(x_train)
x_test=feature.transform(x_test)

# ...............

# movie_rec_dta preparation
movie_rec_model=pickle.load(open("C:\\Users\\Nishanth S\\Desktop\\class\\NLP_project\\movie_rec.sav","rb"))
data_movie=pd.read_csv("C:\\Users\\Nishanth S\\Downloads\\movie_dataset (1).csv")
data_movie=data_movie.iloc[1:]
movie_title=data_movie['title']
movie_titles=[]
for i in movie_title:
    movie_titles.append(i)

def get_title_from_index(index):
    return data_movie[data_movie.index==index]['title'].values[0]
def get_index_from_title(title):
    return data_movie[data_movie.title==title]['index'].values[0]
def get_id_from_index(index):
    return data_movie[data_movie.index==index]['id'].values[0]

def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

# ............



# option menu
with st.sidebar:
    selected=option_menu("NLP (NATURAL LANGUAGE PROCESSOR) APPLICATIONS:",
                         ["SPAM EMAIL DETECTOR","MOVIE RECOMMENDATION SYSTEM"],icons=['envelope-exclamation','film'],default_index=0)

# spam detection

if selected=='SPAM EMAIL DETECTOR':
    st.header("SPAM EMAIL DETECTOR USING NLP")
    st.subheader(" ")
    st.subheader("ENTER THE EMAIL IN BELOW TEXT AREA")
    email=st.text_area("","Enter the email here...")
    email_feature=feature.transform([email])
    spam_pred=''
    if st.button("SUBMIT"):
        prediction=spam_detection_model.predict(email_feature.toarray())
        if prediction[0]==1:
            spam_pred="Entered Email is SPAM!."
        else:    
            spam_pred="Entered Email is NOT SPAM!."
    st.success(spam_pred)

# .................
# Movie recommendation system

if selected=="MOVIE RECOMMENDATION SYSTEM":
    movie=st.selectbox("Type or enter the movie from below",movie_titles,index=0)
    if st.button("RECOMMEND"):
        movies=movie
        movie_index=get_index_from_title(movies)
        similar_movies=list(enumerate(movie_rec_model[movie_index]))
        sorted_similar_movies1=sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]
        recommended_movie_posters = []
        recommended_movie_names=[]
        i=0
        for element in sorted_similar_movies1:
            movie_id=get_id_from_index(element[0])
            recommended_movie_names.append(get_title_from_index(element[0]))
            recommended_movie_posters.append(fetch_poster(movie_id))
            
            # st.success(get_title_from_index(element[0]))
            i=i+1
            if i>4:
                break
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.text(recommended_movie_names[0])
            st.image(recommended_movie_posters[0])
        with col2:
            st.text(recommended_movie_names[1])
            st.image(recommended_movie_posters[1])

        with col3:
            st.text(recommended_movie_names[2])
            st.image(recommended_movie_posters[2])
        with col4:
            st.text(recommended_movie_names[3])
            st.image(recommended_movie_posters[3])
        with col5:
            st.text(recommended_movie_names[4])
            st.image(recommended_movie_posters[4])


# .........




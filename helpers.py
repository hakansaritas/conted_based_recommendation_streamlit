import pandas as pd
import spacy
import re
from string import punctuation
import nltk
nltk.download('vader_lexicon')
from PIL import Image
from IPython.display import HTML
import streamlit as st

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis.sklearn


######################################################

def load_data(dataset_name, data_type):
    if data_type == "csv":
        df = pd.read_csv(dataset_name)
    elif data_type == "xlsx":
        df = pd.read_excel(dataset_name)
    else:
        print("only csv or excel file")

    return df

######################################################
@st.cache(allow_output_mutation=True)
def download_spacy_model(model="en_core_web_sm"):
    spacy.cli.download(model)

######################################################
@st.cache(allow_output_mutation=True)
def load_model(lang_model="en_core_web_sm"):
    nlp = spacy.load(lang_model, disable=["ner", "parser"])
    return nlp
######################################################

def clean_text(text):
    text = text.lower()  # Lowercase text
    text = re.sub(f"[{re.escape(punctuation)}]", "", text)  # Remove punctuation
    text = re.sub(r"[^A-Za-z0-9\s]+", "", text)
    cleaned_text = text.replace("\n", " ").replace("\n" * 2, " ").replace("  ", " ")\
        .replace("   ", " ").replace("    "," ").replace("\n", "", 0)
    #    text = " ".join(text.split())
    return cleaned_text

#####################################################

def tokenize(nlp, cleaned_text):
    doc = nlp(cleaned_text)
    tokens = [token.text for token in doc]
    return tokens

######################################################

def add_stop_words(nlp, stop_list=[]):
    all_stop_words = nlp.Defaults.stop_words
    all_stop_words |= set(stop_list)
    return all_stop_words

######################################################

def remove_stop_words(all_stop_words, tokens):
    sw_tokens = [word for word in tokens if not word in all_stop_words]
    return sw_tokens

######################################################
# If it repeats more than 3 words in a song, it is deleted
def remove_repeat(sw_tokens, n=3):
    repeat_list = []
    nominal_list = []
    for i in sw_tokens:
        if sw_tokens.count(i) > n:
            repeat_list.append(i)
            repeat_list = list(dict.fromkeys(repeat_list))
        else:
            nominal_list.append(i)

        last_list = nominal_list + repeat_list
        return last_list

######################################################

def lemitor(nlp, sw_tokens):
    sw_tokens_nlp = nlp(" ".join(sw_tokens))
    lemma_text = " ".join([token.lemma_ for token in sw_tokens_nlp])
    return lemma_text

######################################################

def text_pos(nlp, lemma_text):
    chunk_text_nlp = nlp(lemma_text)
    for token in chunk_text_nlp:
        print(token.text, token.pos_, )

######################################################

def vectorizer(text):
    tf = TfidfVectorizer(max_df=0.5, min_df=.01)
    tfidf_matrix = tf.fit_transform(text)
    return tfidf_matrix

######################################################

def count_vectorizer(text):
    ct = CountVectorizer(max_df=0.5, min_df=.01)
    count_matrix = ct.fit_transform(text)
    return ct, count_matrix

######################################################

def cosine_similarity_func(tfidf_matrix):
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

######################################################

def sentiment_analyzer(df):
    analyser = SentimentIntensityAnalyzer()

    sentiment_list = []
    sentiment_score_list = []

    for song in df['song_lyric'].values:

        sentiment_score = analyser.polarity_scores(song)

        if sentiment_score['compound'] >= 0.05:
            sentiment_percentage = sentiment_score['compound']
            sentiment = 'Positive'
        elif sentiment_score['compound'] > -0.05 and sentiment_score['compound'] < 0.05:
            sentiment_percentage = sentiment_score['compound']
            sentiment = 'Neutral'
        elif sentiment_score['compound'] <= -0.05:
            sentiment_percentage = sentiment_score['compound']
            sentiment = 'Negative'

        sentiment_list.append(sentiment)
        sentiment_score_list.append((sentiment_percentage) * 100)

    return sentiment_list, sentiment_score_list

##################################################################

def get_image_html(link):
    image_html = f"<img src='{link}' width='100px'>"
    return image_html

##################################################################

def make_clickable(val, name):
    return f'<a href="{val}">{name}</a>'

##################################################################

def play_list(df):
    id_list = df["song_id"].to_list()

    song_html = []

    for i in (id_list):
        song_html.append("https://genius.com/songs/{}/apple_music_player".format(i))

    return song_html

##################################################################

def load_image(image_path):
    image = Image.open(image_path)

    return image

##################################################################

def pipeline(nlp, df, extra_word_list,sentiment=False):

    df["song_lyric_cleaned"] = df["song_lyric"].apply(lambda x: clean_text(x))
    df["song_lyric_cleaned"] = df["song_lyric"].apply(lambda x: clean_text(x))
    df["tokens"] = df["song_lyric_cleaned"].apply(lambda x: tokenize(nlp, x))
    all_stop_words = add_stop_words(nlp, extra_word_list)
    df["sw_tokens"] = df["tokens"].apply(lambda x: remove_stop_words(all_stop_words, x))
    df["lemma_text"] = df["sw_tokens"].apply(lambda x: lemitor(nlp, x))

    if sentiment:
        sentiment_list, sentiment_score_list = sentiment_analyzer(df)
        df['Sentiment'] = sentiment_list
        df['Sentiment_Score'] = sentiment_score_list

    return df

##################################################################

def lda(df,ct,count_matrix, html=False):
    lda = LatentDirichletAllocation(n_components=6, random_state=0)
    lda_matrix = lda.fit_transform(count_matrix).round(5)

    lda_html = pyLDAvis.sklearn.prepare(lda, count_matrix, ct, sort_topics=False)

    df_lda = pd.DataFrame(lda_matrix,index = df["song_lyric"],columns = lda_html.topic_order).add_prefix("topic_")
    df.reset_index(inplace=True, drop=True)
    df_lda.reset_index(inplace=True, drop=True)
    df_topic = df.join(df_lda,how='outer')

    if html:
     pyLDAvis.save_html(lda_html, "lda.html")

    return lda_html, lda_matrix, df_topic

##################################################################

def pairwise_dist(lda_matrix, euclidean=False):

    dists = pairwise_distances(lda_matrix, metric='cosine')

    if euclidean:

        dists = pairwise_distances(lda_matrix, metric='euclidean')

    return dists

##################################################################

def recom_from_text(df, nlp, my_title, my_text, extra_word_list):
    # add your title and text to df

    add_data = {'artist_name': "", 'artist_id': "", 'song_title': my_title, 'song_id': "", 'album_name': "",
                'album_id': "", 'album_date': "", 'artist_image_url': "", 'album_image_url': "",
                'song_views': "", 'song_lyric': my_text}

    df = df.append(add_data, ignore_index=True)

    # clean df
    df["song_lyric_cleaned"] = df["song_lyric"].apply(lambda x: clean_text(x))

    df["tokens"] = df["song_lyric_cleaned"].apply(lambda x: tokenize(nlp, x))

    all_stop_words = add_stop_words(nlp, extra_word_list)

    df["sw_tokens"] = df["tokens"].apply(lambda x: remove_stop_words(all_stop_words, x))

    df["lemma_text"] = df["sw_tokens"].apply(lambda x: lemitor(nlp, x))

    # sentiment analysis
    sentiment_list, sentiment_score_list = sentiment_analyzer(df)

    df['Sentiment'] = sentiment_list

    df['Sentiment_Score'] = sentiment_score_list

    # tfidf - cosine similarity

    tfidf_matrix = vectorizer(df["lemma_text"])

    cosine_sim = cosine_similarity_func(tfidf_matrix)

    indices = pd.Series(df.index, index=df['song_title'])

    song_index = indices[my_title]

    similarity_scores = pd.DataFrame(data=(cosine_sim[song_index]), columns=["score"])

    song_indices = similarity_scores.sort_values("score", ascending=False)[1:5].index

    df_rec = df.iloc[song_indices]

    # make clickable url link with photos
    df_html = df_rec.copy()
    df_html.reset_index(inplace=True)
    df_html.index = df_html.index + 1

    # df_html.drop(["song_lyric", "artist_id", "album_id"], axis=1, inplace=True)
    df_html["artist_image"] = df_html["artist_image_url"].apply(get_image_html)
    df_html["album_image"] = df_html["album_image_url"].apply(get_image_html)

    df_html["album_name"] = df_html.apply(lambda x: make_clickable(x["album_image_url"], x["album_name"]), axis=1)
    df_html["artist_name"] = df_html.apply(lambda x: make_clickable(x["artist_image_url"], x["artist_name"]), axis=1)

    return df_html, HTML(
        df_html[["artist_image", "artist_name", "song_title", "album_name", "album_date", "Sentiment"]].to_html(
            escape=False))

##################################################################

def recom_from_title(df, nlp, song_title, extra_word_list):

    df["song_lyric_cleaned"] = df["song_lyric"].apply(lambda x: clean_text(x))

    df["tokens"] = df["song_lyric_cleaned"].apply(lambda x: tokenize(nlp, x))

    all_stop_words = add_stop_words(nlp, extra_word_list)

    df["sw_tokens"] = df["tokens"].apply(lambda x: remove_stop_words(all_stop_words, x))

    df["lemma_text"] = df["sw_tokens"].apply(lambda x: lemitor(nlp, x))

    # sentiment analysis
    sentiment_list, sentiment_score_list = sentiment_analyzer(df)

    df['Sentiment'] = sentiment_list

    df['Sentiment_Score'] = sentiment_score_list

    # tfidf - cosine similarity

    tfidf_matrix = vectorizer(df["lemma_text"])

    cosine_sim = cosine_similarity_func(tfidf_matrix)

    indices = pd.Series(df.index, index=df['song_title'])

    song_index = indices[song_title]

    similarity_scores = pd.DataFrame(data=(cosine_sim[song_index]), columns=["score"])

    song_indices = similarity_scores.sort_values("score", ascending=False)[1:5].index

    df_rec = df.iloc[song_indices]
    # make clickable url link with photos
    df_html = df_rec.copy()
    df_html.reset_index(inplace=True)
    df_html.index = df_html.index + 1

    # df_html.drop(["song_lyric", "artist_id", "album_id"], axis=1, inplace=True)
    df_html["artist_image"] = df_html["artist_image_url"].apply(get_image_html)
    df_html["album_image"] = df_html["album_image_url"].apply(get_image_html)

    df_html["album_name"] = df_html.apply(lambda x: make_clickable(x["album_image_url"], x["album_name"]), axis=1)
    df_html["artist_name"] = df_html.apply(lambda x: make_clickable(x["artist_image_url"], x["artist_name"]), axis=1)

    return df_html, HTML(df_html[["artist_image", "artist_name", "song_title", "album_name", "album_date", "song_views",
                                  "Sentiment"]].to_html(escape=False))

##################################################################

def lda_recommendation(df, nlp, songid_list, extra_word_list):

    df["song_lyric_cleaned"] = df["song_lyric"].apply(lambda x: clean_text(x))

    df["tokens"] = df["song_lyric_cleaned"].apply(lambda x: tokenize(nlp, x))

    all_stop_words = add_stop_words(nlp, extra_word_list)

    df["sw_tokens"] = df["tokens"].apply(lambda x: remove_stop_words(all_stop_words, x))

    df["lemma_text"] = df["sw_tokens"].apply(lambda x: lemitor(nlp, x))

    # sentiment analysis
    sentiment_list, sentiment_score_list = sentiment_analyzer(df)

    df['Sentiment'] = sentiment_list

    df['Sentiment_Score'] = sentiment_score_list

    ct, count_matrix = count_vectorizer(df["lemma_text"])

    lda_html, lda_matrix, df_topic = lda(df, ct, count_matrix, html=True)

    dists = pairwise_dist(lda_matrix)

    df_dists = pd.DataFrame(data=dists, index=df_topic.song_id, columns=df_topic.song_id)

    songs_summed = df_dists[songid_list].sum(axis=1)

    songs_summed = songs_summed.sort_values( ascending=True)

    song_indices = songs_summed[1:5].index.tolist()

    df_rec = df_topic[df_topic.song_id.isin(song_indices)]

    df_html = df_rec.copy()
    df_html.reset_index(inplace=True)
    df_html.index = df_html.index + 1

    # df_html.drop(["song_lyric", "artist_id", "album_id"], axis=1, inplace=True)
    df_html["artist_image"] = df_html["artist_image_url"].apply(get_image_html)
    df_html["album_image"] = df_html["album_image_url"].apply(get_image_html)

    df_html["album_name"] = df_html.apply(lambda x: make_clickable(x["album_image_url"], x["album_name"]), axis=1)
    df_html["artist_name"] = df_html.apply(lambda x: make_clickable(x["artist_image_url"], x["artist_name"]), axis=1)

    return df_html, HTML(
        df_html[["artist_image", "artist_name", "song_title", "album_name", "album_date", "Sentiment"]].to_html(
            escape=False))
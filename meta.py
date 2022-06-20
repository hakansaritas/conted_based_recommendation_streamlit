code1 = r"""def load_data():
    df = pd.read_csv("hardrock_dataset",index_col=[0])
    return df"""

code2 = r"""
def download_spacy_model(model="en_core_web_sm"):
    lang_model = spacy.cli.download(model)
    return lang_model
    """

code3 = r"""
def load_model():
    nlp = spacy.load()
    return nlp
        """

code4 = r"""
def clean_text(text):
    text = text.lower()  # Lowercase text
    text = re.sub(f"[{re.escape(punctuation)}]", "", text)  # Remove punctuation
    text = re.sub(r"[^A-Za-z0-9\s]+", "", text)
    cleaned_text = text.replace("\n", " ").replace("\n" * 2, " ").replace("  ", " ").replace("   ", " ").replace("    "," ").replace("\n", "", 0)
    return cleaned_text
        """

code5 = r"""
def tokenize(nlp, cleaned_text):
    doc = nlp(cleaned_text)
    tokens = [token.text for token in doc]
    return tokens
        """

code6 = r"""
def add_stop_words(nlp, stop_list=[]):
    all_stop_words = nlp.Defaults.stop_words
    all_stop_words |= set(stop_list)
    return all_stop_words
        """

code7 = r"""
def remove_stop_words(all_stop_words, tokens):
    sw_tokens = [word for word in tokens if not word in all_stop_words]
    return sw_tokens
        """

code8 = r"""
def lemitor(nlp, sw_tokens):
    sw_tokens_nlp = nlp(" ".join(sw_tokens))
    lemma_text = " ".join([token.lemma_ for token in sw_tokens_nlp])
    return lemma_text
        """

code9 = r"""
def vectorizer(dataframe):
    tf = TfidfVectorizer(max_df=0.5, min_df=.01)
    tfidf_matrix = tf.fit_transform(dataframe)
    return tfidf_matrix
        """

code10 = r"""
def cosine_similarity_func(tfidf_matrix):
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim
        """

code11 = r"""
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

    cosine_sim = cosine_similarity(tfidf_matrix)

    indices = pd.Series(df.index, index=df['song_title'])

    song_index = indices[my_title]  # Get the pairwsie similarity scores

    similarity_scores = pd.DataFrame(data=(cosine_sim[song_index]), columns=["score"])

    song_indices = similarity_scores.sort_values("score", ascending=False)[1:5].index

    df_rec = df.iloc[song_indices]
    
    # make clickable url link with photos
    df_html = df_rec.copy()
    df_html.reset_index(inplace=True)
    df_html.index = df_html.index + 1

    df_html.drop(["song_lyric", "artist_id", "album_id"], axis=1, inplace=True)
    df_html["artist_image"] = df_html["artist_image_url"].apply(get_image_html)
    df_html["album_image"] = df_html["album_image_url"].apply(get_image_html)

    df_html["album_name"] = df_html.apply(lambda x: make_clickable(x["album_image_url"], x["album_name"]), axis=1)
    df_html["artist_name"] = df_html.apply(lambda x: make_clickable(x["artist_image_url"], x["artist_name"]), axis=1)

    return df_html, HTML(df_html[["artist_image", "artist_name", "song_title", "album_name", "album_date", "Sentiment"]].to_html( escape=False))
        """



meta1 = "I opened the my own Rock'n'Roll bar which is name Tool-BAR. " \
        " Ofcourse I bought the stools at first before opening the bar:) While I was " \
        "wondering what I could do for the bar, I found a jukebox containing " \
        "only 80's hard rock songs in one of the vintage shops and put it in " \
        "the busiest part of the bar.It is very interesting that this jukebox" \
        " has a strangeness that I discovered later. After someone throws a" \
        " coin into it and shakes the machine, it plays songs with a content" \
        " similar to the one chosen by itself. Much more interestingly, this" \
        " machine has a paper money input, but if you give anything you write" \
        " on paper, such as a song, text, poem, etc., instead of money, the" \
        " jukebox plays songs similar to the content of what you wrote."

meta2 = "Time to open this machine with screw driver and check what it is " \
        "happening. Wooow this machine not old however it more further than us. " \
        "I think my jukebox is coming from Cybertron. OK, If I can, I would want" \
        " to figure out how to work it's song recommendation algorithm. " \
        "I am going to use Data Science."


meta3 = "Content-based methods gives recommendations based on the similarity of two song contents or " \
        "attributes while collaborative methods make a prediction on posible preferences using a matrix " \
        "with ratings on different songs."

meta4 ="Natural language processing is an area of research in computer science and artificial intelligence" \
       " (AI) concerned with processing natural languages such as English or Mandarin. This processing generally " \
       "involves translating natural language into data (numbers) that a computer can use to learn about the world. " \
       "And this understanding of the world is sometimes used to generate natural language text that reflects " \
       "that understanding (Lane et all, 2019)."

meta5 = "TF takes into account how frequently" \
        " a term occurs in a document. Since most of the documents in a text corpus are of" \
        " different lengths, it is very likely that a term would appear more frequently in longer " \
        "documents rather than in smaller ones. This calls for normalizing the frequency of the term " \
        "by dividing it with the count of the terms in the document."

meta6 = "IDF is what does justice to terms that occur not so frequently across documents " \
        " but might be more meaningful in representing the document. It measures the" \
        " importance of a term in a document. The usage of TF only would provide more" \
        " weightage to terms that occur very frequently. As part of IDF, just the opposite is " \
        "done, whereby the weights of frequently occurring terms are suppressed and the " \
        "weights of possibly more meaningful but less frequently occurring terms are " \
        "scaled up."

meta7="👉 User independence: The content-based method only has to analyze the items and a single user’s profile for" \
      " the recommendation, which makes the process less cumbersome. Content-based filtering would thus produce more" \
      " reliable results with fewer users in the system."
meta7_1="👉 Transparency: Collaborative filtering gives recommendations based on other unknown users who have the " \
      "same taste as a given user, but with content-based filtering, items are recommended on a feature-level basis."
meta7_2="👉 No cold start: As opposed to collaborative filtering, new items can be suggested before being rated by a " \
      "substantial number of users (Balu 2019)."

meta8="👉 Limited content analysis: If the content doesn’t contain enough information to discriminate the items" \
      " precisely, the recommendation itself risks being imprecise."
meta8_1="👉 Over-specialization: Content-based filtering provides a limited degree of novelty since it has to match " \
      "up the features of a user’s profile with available items. In the case of item-based filtering, only item " \
      "profiles are created and users are suggested items similar to what they rate or search for, instead of their" \
      " past history. A perfect content-based filtering system may suggest nothing unexpected or surprising (Balu 2019)."

meta9 = "In natural language processing, Latent Dirichlet Allocation (LDA) is a generative statistical" \
        " model that explains a set of observations through unobserved groups, and each group explains why" \
        " some parts of the data are similar. LDA is an example of a topic model. In this, observations " \
        "(e.g., words) are collected into documents, and each word's presence is attributable to one of the " \
        "document's topics. Each document will contain a small number of topics. " \
        "(wikipedia: https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)"



#### to get requirements.txt of project ####
# pip install pipreqs
# c:\Users\your_user_name\anaconda3\Lib\site-packages>pipreqs  --encoding utf-8 "your project directory" --force

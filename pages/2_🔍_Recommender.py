import streamlit as st
import streamlit.components.v1 as components
import helpers


st.title("📣 RECOMMENDATION ENGINE ")
st.write( "There are three kind of recommendation selection engines.")
st.write("1 - Select one song title from database and take nearest songs.")
st.write("2 - Type what ever you want with its title and take nearest songs.")
st.write("3 - Select three song titles from database and take nearest songs.")

with st.sidebar:
    data_load_state = st.text('Loading data...')
    df = helpers.load_data()
    data_load_state.text('Loading data...done!')

    data_load_state = st.text('Loading spacy model...')
    helpers.download_spacy_model()
    data_load_state.text('Loading spacy model...done!')

    data_load_state = st.text('Loading nlp...')
    nlp = helpers.load_model()
    data_load_state.text('Loading nlp...done!')

extra_word_list = ['aaaaaah','aaaaah','aaaaaahhhhh', 'aaahaaahaaah', 'aaahahah', 'ah','ahah', 'ahahah',
                   'ahahahah', 'ahahahahahah', 'ahh', 'ahhhahah', 'ahhhahhhahhh','ai', 'ay','ayayayayayay',
                   'd', 'do','get', 'gon', 'gonna', 'got', 'gotta', 'ha','haa','haaaa', 'haha','hahaha',
                   'hahahaha','hahahahaha','hahahahahaha','heeheheh','heh','hey','hoo','ll','la', 'lalalala',
                   'lalala','lalalalalalalala', 'let', 'm','mm','mmmm', 'na','nana', 'nanana','nt', 'oo',
                   'oh', 'oooh','ooooh','ohoh', 'ohohoh','ohohohoh', 'ohuhoh', 'oi', 'ooh', 'oooo','pron',
                   's', 'ta', 'to', 'too','uh', 'uhh', 'uhhhuhh', 'uhhhuhhhuhhh', 'uhhhuhuh', 'ya', 'yea',
                   'yeah','yeahyeah', 'yeahyeahyeah','yup','yyo','ve']

with st.container():
    st.subheader("1 - Recommendation by Song Title")
    with st.expander("Choose Song Title", expanded=False):
        artist = df["artist_name"].tolist()
        songs = df["song_title"].tolist()
        options = list(zip(artist,songs ))
        options = sorted(options)
        chooser = st.selectbox("Artist-Song Title", options)
        song_title = chooser[1]

    if st.button('get recommendation',key=1):
        data_load_state = st.text('please wait ⏱️...it takes approximately 2-3 minutes')
        df_html, df_recom = helpers.recom_from_title(df,nlp,song_title, extra_word_list)
        data_load_state.text("let's ROCK!")
        song_html = helpers.play_list(df_html)

        st.subheader("tracks (only 30 sec)")
        for song_url in song_html:
            components.iframe(song_url, height=200, scrolling=True)

        st.subheader("Info")
        st.write(df_recom)

    st.write("--------------------------------------------------------------------")

    st.subheader("2 - Recommendation by Your Text")
    with st.expander("Enter your title and text", expanded=False):

        my_title = st.text_input("your title")
        st.write(my_title)

        my_text = st.text_area("type something")
        st.write(my_text)

    if st.button('get recommendation',key=2):
        data_load_state = st.text('please wait ⏱️...it takes approximately 2 minutes')
        df_html, df_recom = helpers.recom_from_text(df, nlp, my_title, my_text, extra_word_list)
        data_load_state.text("let's ROCK!")
        song_html1 = helpers.play_list(df_html)

        st.subheader("tracks (only 30 sec)")
        for song_url in song_html1:
            components.iframe(song_url, height=50, scrolling=True)

        st.subheader("Info")
        st.write(df_recom)

    st.write("--------------------------------------------------------------------")

    st.subheader("3 - LDA Recommendation by  3 Songs")
    with st.expander("Choose 3 Song Titles", expanded=False):
        artist = df["artist_name"].tolist()
        songs = df["song_title"].tolist()
        id = df["song_id"].tolist()
        options = list(zip( artist,songs, id))
        options = sorted(options)
        chooser1 = st.selectbox("Artist-Song Title-Song ID ", options, index = 0,key=1)
        chooser2 = st.selectbox("Artist-Song Title-Song ID ", options, index =50, key=2)
        chooser3 = st.selectbox("Artist-Song Title-Song ID ", options, index = 100, key=3)
        songid_list = [chooser1[2],chooser2[2],chooser3[2]]
        st.text(songid_list)

    if st.button('get recommendation',key=3):
        data_load_state = st.text('please wait ⏱️...it takes approximately 1-2 minutes')
        df_html, df_recom = helpers.lda_recommendation(df,nlp, songid_list, extra_word_list)
        data_load_state.text("let's ROCK!")
        song_html2 = helpers.play_list(df_html)

        st.subheader("tracks (only 30 sec)")
        for song_url in song_html2:
            components.iframe(song_url, height=50, scrolling=True)

        st.subheader("Info")
        st.write(df_recom)



        # with st.container():
        #     songs = df_html["song_title"].tolist()
        #     lyric = df_html["song_lyric"].tolist()
        #     song_title_lyric = st.radio("LYRICS", songs)

            # if song_title_lyric == songs[0]:
            #     st.text(lyric[0])
            # elif song_title_lyric == songs[1]:
            #     st.write(lyric[1])
            # elif song_title_lyric == songs[2]:
            #     st.write(lyric[2])
            # elif song_title_lyric == songs[3]:
            #     st.write(lyric[3])






import streamlit as st
from helpers import load_image
import meta
st.set_page_config(layout="wide", page_title="HARD ROCK Song Rec", page_icon=":guitar:")

st.sidebar.title("Contribution")
st.sidebar.info("This is an open source project and you are very"
                " welcome to contribute your comments, questions "
                "resources and apps to the source code")

st.sidebar.title("Contact")
st.sidebar.info(
    """
    This project is created by Hakan SARITAS in scope of "Veri Bilim Okulu - Data Sicence
     and Machine Learning Bootcamp". I am junior Data Scientist, so helpful 
     hints and suggestions are welcome. You can learn more about me at 
    [E-mail]: saritas_hakan@yahoo.com
    [LinkedIn](https://www.linkedin.com/in/hakansaritas/) | [GitHub](https://github.com/hakansaritas) |
     [Medium](https://hakansaritas.medium.com/) | 
    """
)

st.title ("🎧 CONTENT BASED SONG RECOMMENDATION ")
st.text("\n")
st.text("\n")

col1, mid, col2= st.columns([1, 5, 20])
with col1:
    st.image(load_image("./pic/jukebox.jpg"),width=150)
with col2:
    st.subheader("Story 🎬")
    st.write(meta.meta1)




st.write(meta.meta2)

st.text("\n")
st.text("\n")
st.subheader("AIM-Pipeline 🎯")
with st.container():
    with st.expander("1 - Create and load a database including hard rock artist names, songs titles,lyrics.", expanded=False):
        code =(meta.code1)
        st.code(code, language='python')

    with st.expander("2 - Create and load a English language model using Spacy .", expanded=False):
        code = (meta.code2)
        st.code(code, language='python')
        code = (meta.code3)
        st.code(code, language='python')

    with st.expander("3 - Clear the lyrics from punctuations,spaces,...", expanded=False):
        code = (meta.code4)
        st.code(code, language='python')

    with st.expander("4 - Separating each lyric into words - tokenize. ", expanded=False):
        code = (meta.code5)
        st.code(code, language='python')

    with st.expander("5 - Removing meaningless entries like am,is,are...", expanded=False):
        code = (meta.code6)
        st.code(code, language='python')
        code =(meta.code7)
        st.code(code, language='python')

    with st.expander("6 - lemmatization of the words to find their roots and reduce count of the words.", expanded=False):
        code = (meta.code8)
        st.code(code, language='python')

    with st.expander("7 - Vectorizing of the words using TF-IDF for the computer to understand ", expanded=False):
        code = (meta.code9)
        st.code(code, language='python')

    with st.expander("8 - Find the content based similarities between two lyrics using Cosine Similarity", expanded=False):
        code = (meta.code10)
        st.code(code, language='python')

    with st.expander("9 - Get songs recommended by computer.", expanded=False):
        code = (meta.code11)
        st.code(code, language='python')

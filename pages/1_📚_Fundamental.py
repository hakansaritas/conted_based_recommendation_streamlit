import streamlit as st
import meta
from helpers import load_image

st.subheader("What is Content Based Song Recommendation?")
st.write(meta.meta3)
with st.container():
    col1, col2, col3, = st.columns([1,2,1])
    with col2:
        st.image(load_image("./pic/cont_bas_rec.jpg"),width=600)
        st.caption("Example of very crude content-based recommendation for `Ex Machine` movie (Falk 2019)")

st.write("-------------------------------------------------------")
with st.container():
    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader("Advantages of Content-Based Filtering 👍")
        st.write(meta.meta7)
        st.write(meta.meta7_1)
        st.write(meta.meta7_2)

    with col2:
        st.subheader("Disadvantages of Content-Based Filtering 👎")
        st.write(meta.meta8)
        st.write(meta.meta8_1)
st.write("-------------------------------------------------------")
with st.container():
    st.subheader("What is NLP ?")
    st.write(meta.meta4)

    st.image(load_image("./pic/NLP_ex.jpg"),width=600)
    st.caption("Categorized NLP applications (Lane et all, 2019)")
st.write("-------------------------------------------------------")
with st.container():
    st.subheader("What is TF ?")
    st.write(meta.meta5)
st.write("-------------------------------------------------------")
with st.container():
    st.subheader("What is IDF ?")
    st.write(meta.meta6)

    st.image(load_image("./pic/tf-idf.jpg"),width=600)
    st.caption("TF-IDF formulas https://machinelearningflashcards.com/")
st.write("-------------------------------------------------------")
with st.container():
    st.subheader("What is LDA ?")
    st.write(meta.meta9)

    st.image(load_image("./pic/lda.jpg"),width=600)
    st.caption("TF-IDF formulas https://machinelearningflashcards.com/")
    st.caption("topic model. Each topic is defined by a list of words "
               "and their respective probability of being drawn. "
               "A document can be described by selecting topics, "
               "using a formulaof how large a percentage of the time "
               "you should draw from each topic (Falk 2019).")



st.subheader("🔖 References")
st.markdown("Kim Falk (2019). Recommender Systems. Manning Publications Co.")
st.markdown("Hobson Lane, Cole Howard, Hannes Max Hapke (2019). Natural Language Processing IN ACTION.Manning Publications Co. ")
st.markdown("TF-IDF formulas https://machinelearningflashcards.com/")
st.markdown("wikipedia: https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation")
st.markdown("Bindhi Balu (2019). Content-Based Recommendation System. Medium. https://medium.com/@bindhubalu/content-based-recommender-system-4db1b3de03e7")
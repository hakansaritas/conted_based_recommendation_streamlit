
import streamlit.components.v1 as components
import streamlit as st

st.subheader("Latent Dirichlet allocation")
st.write("If you enter a high alpha, then you’ll distribute each document "
         "over many topics; low alpha distributes only a few topics. "
         "The advantage with high alpha is that documents seem to be more similar,"
         " while if you have specialized documents, then a low alpha will keep them"
         " divided into few topics (Falk 2019)")

           
 HtmlFile = open("./pages/lda.html", 'r', encoding='utf-8')
 source_code = HtmlFile.read()
 print(source_code)
 components.html(source_code,height = 800, width = 1300)

#pip install spacy-streamlit
import spacy_streamlit

models = ["en_core_web_sm", "en_core_web_md"]
default_text = "Text to analyse by Spacy"
spacy_streamlit.visualize(models, default_text)
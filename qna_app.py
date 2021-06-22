import joblib
import requests
import streamlit as st
st.set_option('deprecation.showfileUploaderEncoding',False)
st.title('QnA App')
st.text('Ask questions relevant to Earnings call')

@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load('/content/models/bert_qa_custom.joblib')
    return model

with st.spinner('Loading Model Into Memory...'):
    model = load_model()

text = st.text_input("Enter your questions here..")

if text:
    st.write("Response :")
    with st.spinner("Searching for answers.."):
        prediction = model.predict(text)
        st.write('answer: {}'.format(prediction[0]))
        st.write('title: {}'.format(prediction[1]))
        st.write('paragraph: {}'.format(prediction[2]))
        
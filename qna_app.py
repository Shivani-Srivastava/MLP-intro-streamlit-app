import joblib
import streamlit as st
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding',False)
st.title('QnA App')
st.text('Ask questions relevant to Earnings call')

@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load('bert_qa_custom.joblib')
    return model
@st.cache(allow_output_mutation=True)
def load_data():
  df = pd.read_csv('df.csv')
  return df

def main():
  
  with st.spinner('Loading Model Into Memory...'):
    model = load_model()
  text = st.text_input("Enter your questions here..")
  n_rtn = st.sidebar.slider("Number of returned predictions",1,6,1)
  st.markdown('Data used for model training')
  st.write(load_data())
  if text:
    
    with st.spinner("Searching for answers.."):
        prediction = model.predict(text,n_predictions = n_rtn)

    st.write("Response :")
    for p in prediction:
      st.write('--------------------------------------')
      st.write('answer: {}'.format(p[0]))
      st.write('title: {}'.format(p[1]))
      st.write('paragraph: {}'.format(p[2]))
      st.write('retriever_score_weight: {}'.format(round(p[3],3)))
      st.write('--------------------------------------')


if __name__=='__main__':
  main()

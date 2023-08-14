# UI Library
import streamlit as st 
from streamlit_option_menu import option_menu

# Base Libraries
import pandas as pd
import joblib
from keras.models import load_model
import time

from keras_preprocessing.sequence import pad_sequences

############# Data and Saved model ##############
data = pd.read_csv("train.csv")
model = load_model("tweet_dection.h5")
tokenizer = joblib.load("token.pkl")


########## UI Background ###################

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("data:image/jpeg;base64,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");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )


add_bg_from_url()

labels = {0:'bot', 1:'human'}
def predict(text):
    import time
    start_at = time.time()
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]),maxlen=30)
    score = model.predict([x_test])
    print(score[0][0])
    if score[0][0]>0.5:
        c1 = 1
    else:
        c1 = 0
    label = labels[c1]
    #print(st.write(x_test))
    return {"Prediction":c1, "label":label}


########################################### UI ##########################
with st.sidebar:
    choose = option_menu("Tweet Detection Model", ["Project Info", "Data Studied", "Detection", "About Me"],
                         icons=['house', 'table', "tags-fill", 'person'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "black", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "skyblue"},

    }
    )

if choose == "Project Info":
    st.markdown(""" <style> .font {
        font-size:30px ; font-family: 'Cooper Black'; color: #FF9633;}
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">About the Project:</p>', unsafe_allow_html=True)    

    st.markdown(""" <style> .para {
        font-size:20px ; font-family: 'Calibri'; color: black; text-align: 'justify'}
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="para">A web app to identify whether a tweet is tweeted by humans or bot.</p>', unsafe_allow_html=True)

    st.markdown(""" <style> .font {
        font-size:20px ; font-family: 'Cooper Black'; color: #FF9633;}
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font"><b>INTRODUCTION:</b></p>', unsafe_allow_html=True)  

    st.markdown(""" <style> .para {
        font-size:20px ; font-family: 'Calibri'; color: black; text-align: 'justify'}
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="para">Twitter is a popular online social networking and micro-blogging tool, which was released in 2006. Remarkable simplicity is its distinctive feature. Its community interacts via publishing text-based posts, known as tweets. The tweet size is limited to 140 characters.</p>', unsafe_allow_html=True)    

    st.markdown(""" <style> .font {
        font-size:20px ; font-family: 'Cooper Black'; color: #FF9633;}
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font"><b>Keywords:</b></p>', unsafe_allow_html=True)  

    st.markdown(""" <style> .para {
        font-size:20px ; font-family: 'Calibri'; color: black; text-align: 'justify'}
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="para">Automatic Identification:- Bot or Human</p>', unsafe_allow_html=True)

    st.markdown(""" <style> .font {
        font-size:20px ; font-family: 'Cooper Black'; color: #FF9633;}
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font"><b>Model:</b></p>', unsafe_allow_html=True) 

    st.markdown(""" <style> .para {
        font-size:20px ; font-family: 'Calibri'; color: black; text-align: 'justify'}
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="para"><b>NLP with LSTM </b></p>', unsafe_allow_html=True)

elif choose == "Data Studied":
    st.markdown(""" <style> .font {
        font-size:30px ; font-family: 'Cooper Black'; color: #FF9633;}
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Data Info:</p>', unsafe_allow_html=True)    

    st.markdown(""" <style> .para {
        font-size:20px ; font-family: 'Calibri'; color: black; text-align: 'justify'}
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="para">Data is taken from Kaggle.</p>', unsafe_allow_html=True)

    st.markdown(""" <style> .para {
        font-size:20px ; font-family: 'Calibri'; color: black; text-align: 'justify'}
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="para">Collected data is huge and not having duplicates,removed unwanted columns for the analysis. These were handled</p>', unsafe_allow_html=True)
    
    st.markdown(""" <style> .para {
        font-size:20px ; font-family: 'Calibri'; color: black; text-align: 'justify'}
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="para"><b>Sample of Raw Data Collected:</b></p>', unsafe_allow_html=True)
    st.dataframe(data.head())
    st.markdown(""" <style> .para {
        font-size:20px ; font-family: 'Calibri'; color: black; text-align: 'justify'}
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="para">From the above data account.type is our Output column for analysis and prediction.</p>', unsafe_allow_html=True)
    st.markdown(""" <style> .para {
        font-size:20px ; font-family: 'Calibri'; color: black; text-align: 'justify'}
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="para">Removed unwanted columns and renamed the target variabe as Category.</p>', unsafe_allow_html=True)

elif choose == "About Me":
    st.markdown(""" <style> .font {
        font-size:30px ; font-family: 'Cooper Black'; color: #FF9633;}
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">About Me:</p>', unsafe_allow_html=True)    
    st.markdown('<p class="para">Name: T.Gouri Sankar</p>', unsafe_allow_html=True)
    st.markdown('<p class="para">Email: gourisankarterlada@gmail.com</p>', unsafe_allow_html=True)

    st.markdown('<p class="para">GitHub:-  https://github.com/TGouriSankar</p>', unsafe_allow_html=True)
    st.markdown('<p class="para">Kaggle:- https://www.kaggle.com/tgourisankar </p>', unsafe_allow_html=True)
    st.markdown('<p class="para">Linkedin:-  https://www.linkedin.com/in/t-gouri-sankar-7372921b8/</p>', unsafe_allow_html=True)
    st.markdown('<p class="para">Naukri:-  https://www.naukri.com/mnjuser/homepage/</p>', unsafe_allow_html=True)
    st.markdown('<p class="para">Phone No: 7978999031</p>', unsafe_allow_html=True)

    st.markdown(""" <style> .font {
        font-size:20px ; font-family: 'Cooper Black'; color: #FF9633;}
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">["Please contact me for any queries on the project"]</p>', unsafe_allow_html=True)


elif choose == "Detection":
    ##### Prediction on Single review Code #####

    st.markdown(""" <style> .para {
        font-size:20px ; font-family: 'Calibri'; color: black; text-align: 'justify'}
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="para"><b>Detection on given tweet:</b></p>', unsafe_allow_html=True)

    text = st.text_area("Enter A Twitter Handle")

    if st.button("detect"):
        st.header(predict(text))
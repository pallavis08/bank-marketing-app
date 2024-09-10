import pickle
import streamlit as st
from PIL import Image
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier

# st.title(':blue[Target the right customer!]')
st.set_page_config(layout = "wide")
title = '<p style="font-family:sans-serif; text-align: center; color:#66F6F1; font-size: 40px; font-weight: bold;">Target the right customer!</p>'
st.markdown(title, unsafe_allow_html=True)
st.markdown('####')
img1 = Image.open('marketing.jpg')
img1_1 = img1.resize((600, 400))
img2 = Image.open('bank_1.jpg')
img2_1 = img2.resize((600, 400))
col1, col2, col3, col4, col5 = st.columns([1,6,0.05,6,1])
with col1:
    st.write("")
with col2:
    st.image(img1_1, width=500)
with col3:
    st.write("")
with col4:
    st.image(img2_1, width=500)
with col5:
    st.write("")
st.markdown('######')
st.caption('This product is build for the marketing team members, working in an Awesome bank. It filter out the customers \
           who are most likely to take the subscription of a Term deposit. User can upload a csv file with the required data \
           (details on the left panel). List of customer ids and prediction will be displayed. Customers that are the good \
           match for the Term deposit campaign will be scored 1. This list can also be downloaded as a csv file')

with st.sidebar:
    st.header('Data Requirements')
    st.caption('To determine which customer to contact for a Term Deposit Subscription, upload a csv file with following columns:')
    with st.expander('Name and format'):
        st.markdown('1 - customer_id (string)')
        st.markdown('2 - age (numeric)')
        st.markdown('3 - job : type of job (categorical)')
        st.markdown('4 - marital : marital status (categorical)')
        st.markdown('5 - education (categorical)')
        st.markdown('6 - default: has credit in default? (categorical)')
        st.markdown('7 - housing: has housing loan? (categorical)')
        st.markdown('8 - loan: has personal loan? (categorical)')
        st.markdown('9 - contact: contact communication type (categorical)')
        st.markdown('10 - month: last contact month of year (categorical)')
        st.markdown('11 - day_of_week: last contact day of the week (categorical)')
        st.markdown('12 - duration: last contact duration, in seconds (numeric)')
        st.markdown('13 - campaign: number of contacts during this campaign (numeric)')
        st.markdown('14 - pdays: number of days since last contact from a previous campaign (numeric)')
        st.markdown('15 - previous: number of contacts before this campaign (numeric)')
        st.markdown('16 - poutcome: outcome of the previous campaign (categorical)')
        st.markdown('17 - emp.var.rate: employment variation rate - quarterly indicator (numeric))')
        st.markdown('18 - cons.price.idx: consumer price index - monthly indicator (numeric)')
        st.markdown('19 - cons.conf.idx: consumer confidence index - monthly indicator (numeric))')
        st.markdown('20 - euribor3m: euribor 3 month rate - daily indicator (numeric)')
        st.markdown('21 - nr.employed: number of employees - quarterly indicator (numeric)')        
    st.divider()
    st.caption('Developed by Pallavi Srivastava')

if 'clicked' not in st.session_state:
    st.session_state.clicked = {1:False}

def clicked(button):
    st.session_state.clicked[button] = True

st.button("Let's get started", on_click = clicked, args = [1])

if st.session_state.clicked[1]:
    uploaded_file = st.file_uploader("Choose a file", type='csv')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, low_memory = True)
        # Display a sample of data
        st.header(':blue[Uploaded data sample]')
        st.write(df.head())
        # Load in model, scaler and columns used for training
        model = pickle.load(open('best_model_gbc2.pkl','rb'))
        scaler = pickle.load(open('scaler.sav', 'rb'))
        cols_training_obj = pickle.load(open('cols_training.sav', 'rb'))
        cols_training = (cols_training_obj).to_list()
        #Encoding categorical variables
        categorical_vars = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
        df_cat=pd.get_dummies(df[categorical_vars], drop_first = False, dtype=int)
        # Scaling numerical variables
        numerical_vars =  ['campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx','cons.conf.idx', 'nr.employed','age','euribor3m']
        scaled = scaler.transform(df[numerical_vars])
        df_num = pd.DataFrame(scaled, columns=numerical_vars)
        # Combing both
        X = pd.concat([df_cat, df_num], axis = 1)
        # Taking same columns as used in training
        X = X[[c for c in X.columns if c in cols_training]]
        missing_cols = [x for x in cols_training if x not in X.columns.to_list()]
        for col in missing_cols:
            X[col] = 0
        # Reordering columns
        X_n = X.reindex(columns=cols_training)
        pred = model.predict(X_n)
        df['prediction'] = pred
        df_res = df[['customer_id', 'prediction']]
        st.header(':blue[Predicted values]')
        st.write(df_res.head())

        pred = df_res.to_csv(index=False).encode('utf-8')
        st.download_button('Download prediction',
                        pred,
                        'prediction.csv',
                        'text/csv',
                        key='download-csv')

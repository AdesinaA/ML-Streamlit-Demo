import streamlit as st
from utils import PrepProcesor, columns 

import numpy as np
import pandas as pd
import joblib

model = joblib.load('xgbpipe.joblib')
st.title('Did the Passengers survive? :ship:')

passengerid = st.text_input("Input Passenger ID", '2580') 
pclass = st.selectbox("Choose class", [1,2,3])
name  = st.text_input("Input Passenger Name", 'Adesina Abdulrahman')
sex = st.select_slider("Choose sex", ['male','female'])
age = st.slider("Choose age",0,100)
sibsp = st.slider("Choose siblings",0,10)
parch = st.slider("Choose parch",0,2)
ticket = st.text_input("Input Ticket Number", "24432") 
fare = st.number_input("Input Fare Price", 0,1000)
cabin = st.text_input("Input Cabin", "C52") 
embarked = st.select_slider("Did they Embark?", ['S','C','Q'])

def predict(): 
    row = np.array([passengerid,pclass,name,sex,age,sibsp,parch,ticket,fare,cabin,embarked]) 
    X = pd.DataFrame([row], columns = columns)
    prediction = model.predict(X)
    if prediction[0] == 1: 
        st.success('Passenger Survived :thumbsup:')
    else: 
        st.error('Passenger did not Survive :thumbsdown:') 

trigger = st.button('Predict', on_click=predict)


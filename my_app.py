# Streamlit Documentation: https://docs.streamlit.io/


import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image  # to deal with images (PIL: Python imaging library)





img = Image.open("images.png")
st.image(img, width=1920)

st.header('This is a car price prediction app.')

st.text("Select your car features on side bar and app will return your car price.")



st.sidebar.header("Chose your car features...")



# To load machine learning model
import pickle
filename = "autoscout_aws_lasso.pkl"
model = pickle.load(open(filename, "rb"))

df = pd.read_csv('autoscout24_dummies2_aws.csv')

make_model_u = df['make_model'].drop_duplicates()
body_type_u = df['body_type'].drop_duplicates()
gearbox_u = df['gearbox'].drop_duplicates()
fuel_type_u = df['fuel_type'].drop_duplicates()
engine_size_cat_u = df['engine_size_cat'].drop_duplicates()
type_u = df['type'].drop_duplicates()

# To take feature inputs
make_model = st.sidebar.selectbox("Select your car model...:", make_model_u)
body_type = st.sidebar.selectbox("Select your car's body type...:", body_type_u)
power = st.sidebar.number_input("Select the power of your car...:", min_value=12, max_value=450, step= 1)
gearbox = st.sidebar.selectbox("Select the gear box of your car...:", gearbox_u)
doors = st.sidebar.slider("Select your car door count...:",min_value=3, max_value=5, step=1)
fuel_type = st.sidebar.selectbox("Select the fuel type of your car...:", fuel_type_u)
mileage = st.sidebar.number_input("Select your car mileage...:",min_value=0, max_value=263000)
age = st.sidebar.slider("Select your car age...:",min_value=0, max_value=21, step=1)
engine_size_cat = st.sidebar.selectbox("Select the engine size category of your car...:", engine_size_cat_u)
type = st.sidebar.selectbox("Select the type of your car...", type_u)

# Create a dataframe using feature inputs
new_sample = {
    'make_model' : make_model,
    'body_type'  : body_type,
    'power'      : power,
    'gearbox'    : gearbox,
    'doors'      : doors,
    'fuel_type'  : fuel_type,
    'mileage'   : mileage,
    'age'       : age,
    'engine_size_cat' : engine_size_cat,
    'type'      : type
}

df = pd.DataFrame.from_dict([new_sample])
st.table(df)

# Prediction with user inputs
predict = st.button("Predict")
result = model.predict(df)
if predict :
    st.success(result[0])
    st.balloons()



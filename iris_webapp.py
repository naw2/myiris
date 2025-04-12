import streamlit as st
import joblib
import numpy as np
st.set_page_config(page_title = 'My Iris App')
st.title('Iris Flower Classification')
img = 'iris.png'
st.image(img,caption = 'Iris Flower')
@st.cache(allow_output_mutation = True) 
def get_model():
    return joblib.load('iris.joblib')
spl = st.text_input("Enter Sepal Length: ", "")
spw = st.text_input("Enter Sepal Width: ", "")
pel = st.text_input("Enter Petal Length: ", "")
pew = st.text_input("Enter Petal Width: ", "")
if st.button("Classify Flower"):
    values = [spl,spw,pel,pew]
    float_values = []
    for x in values:
        float_values.append(float(x))
    input_values = np.asarray(float_values).reshape(1,-1)
    model = get_model()
    pred=model.predict(input_values)
    pred = int(pred[0])
    if pred == 0:
        st.write("Flower is Sentosa")
    elif pred == 1:
        st.write("Flower is Versicolor")
    elif pred == 2:
        st.write("Flower is Verginica")
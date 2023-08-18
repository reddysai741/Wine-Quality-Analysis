import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


background_style = """
    <style>
        body {
            background-image: url("C:\sai\wine1.jpeg");
            background-size: cover;
        }
    </style>
"""

st.markdown(background_style, unsafe_allow_html=True)


st.write("""
# Wine Quality Prediction App
This app predicts the ***Wine Quality*** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
        fixed_acidity = st.sidebar.slider('fixed acidity', 4.6, 15.9, 8.31)
        volatile_acidity = st.sidebar.slider('volatile acidity', 0.12,1.58 , 0.52)
        citric_acid = st.sidebar.slider('citric acid', 0.0,1.0 , 0.5)
        chlorides = st.sidebar.slider('chlorides', 0.01,0.6 , 0.08)
        total_sulfur_dioxide=st.sidebar.slider('total sulfur dioxide', 6.0,289.0 , 46.0)
        alcohol=st.sidebar.slider('alcohol', 8.4,14.9, 10.4)
        sulphates=st.sidebar.slider('sulphates', 0.33,2.0,0.65 )
        data = {'fixed_acidity': fixed_acidity,
                'volatile_acidity': volatile_acidity,
                'citric_acid': citric_acid,
                'chlorides': chlorides,
              'total_sulfur_dioxide':total_sulfur_dioxide,
              'alcohol':alcohol,
                'sulphates':sulphates}
        features = pd.DataFrame(data, index=[0])
        return features
df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

#
data=pd.read_excel("C:\sai\Wine_data.xlsx")
X =np.array(data[['fixed acidity', 'volatile acidity' , 'citric acid' , 'chlorides' , 'total sulfur dioxide' , 'alcohol' , 'sulphates']])
Y = np.array(data['quality'])


rfc= RandomForestClassifier()
rfc.fit(X, Y)

st.subheader('Wine quality labels and their corresponding index number')
st.write(pd.DataFrame({
   'wine quality': [3, 4, 5, 6, 7, 8 ]}))

prediction = rfc.predict(df)
prediction_proba = rfc.predict_proba(df)

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)

fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=Y, cmap='viridis')
ax.set_xlabel('Fixed Acidity')
ax.set_ylabel('Volatile Acidity')
ax.set_title('Wine Quality Scatter Plot')
st.pyplot(fig)

fig,ax=plt.subplots(ncols=6,nrows=2,figsize=(10,10))
index=0
ax=ax.flatten()
for col, value in data.items():
  if col != 'type':
    sns.boxplot(y=col, data=data, ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5,w_pad=0.7,h_pad=5.0)
st.pyplot(fig)

from scipy.stats import skew
features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
fig, axes = plt.subplots(3, 4, figsize=(20,20))
for i, ax in enumerate(axes.flat):
    sns.barplot(data = data, y = features[i], x = 'quality', ax = ax)

    sk = skew(data[features[i]])

    print("Feature {} has skewness {}".format(features[i], sk))

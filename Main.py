import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier



page_bg_img = """

 <style>
            [data-testid="stAppViewContainer"] {
                background-image: url("data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxITEhMSExIWFRUWFR0aFxgWGBgZGBYfGR0ZGBYYGBcYHSggGBooGxUYITEiJikvLi4uFx8zODMsNygtLisBCgoKDg0OGhAQGy0lICUtNS8tLTctLS0tLy0tKy8tLS0vLS0rLSstLS0tLSs1LS0tLS0tLS0tLS0tLS0tLS0tLf/AABEIAJ8BPgMBIgACEQEDEQH/xAAcAAEAAwEBAQEBAAAAAAAAAAAABQYHBAgDAgH/xABOEAABAwIDAwcGCQYMBwAAAAABAAIDBBEFEiEGMUEHEyJRYXGBIzKRobGyFDNSYnKCwcLRJEJzkrPwCCU0Q1NjZHST0tPhFRYYRFWDov/EABoBAQADAQEBAAAAAAAAAAAAAAABAgMEBQb/xAAtEQACAgECBAQFBQEAAAAAAAAAAQIRAyExBBJBUTJxsfATFCJhwQUzgeHxkf/aAAwDAQACEQMRAD8A3FERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAZjyhY5VQxvdFMWESuAyusbAkAWtbS3qVq5OcVkqcOpppXF8jmkPcbAkse5hJtp+aqNypwN5kuAsfhD9ePnOVt5Iz/ABTS90n7WRa5FojLG3bLgiIsjUpXKvjs9JSRvp5ObkfMG3ytdpke4jpAgeaNbKn8lG21fU15gqajnWGJ5AyRt6TS22rGg7i5WPlsiLqOFot/KRv7I5T4LPuRKO2KA/1En3PQgPQaIiALAdutucRirJGR1L42NmkaGhrLAMcWt3t10F/Fb8vMPKQb18v94m9TyPsWeR1R18JFSbtdPyehtj62SehpJpTeSSBjnmwGYloJdYAAX36damFBbBn+LaH+6xe41TqujlluwozFp3NLQHW3qTUFtNM5oaWi50t4kBG6VkJWSVBISNTddajcDkc5gLhY21t2EhSSRdqw1QUNtji76SjnqWNa50bQWh18pJIaL21O/dxUyqjysi+E1X/r/ax2KPYvBXJJ9yk4Lyp10lTSxyR02SaZrHWa9rmhzg24JkNjrxButkXl7BG/ldELk2qofXKx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");
                background-size: 180%;
                background-position: top left;
                background-repeat: no-repeat;
                background-attachment: local;
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

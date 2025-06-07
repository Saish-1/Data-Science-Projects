import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings 
warnings.filterwarnings('ignore')

st.title("Survival Prediction of Titanic")
st.subheader("Find if a person survived or not in the Titanic tragedy")

st.sidebar.header("User Input Parameters")

def user_input_features():
	SEX = st.sidebar.selectbox('Gender (0-Female, 1-Male)',('1','0'))
	AGE = st.sidebar.number_input("Enter Age",min_value=1.0,max_value=100.0,step=1.0)
	PCLASS = st.sidebar.selectbox('Pclass',('1','2','3'))
	SIBSP = st.sidebar.number_input('Enter no. of sibilings and spouse',min_value=0.0,max_value=10.0,step=1.0)
	PARCH = st.sidebar.number_input('Enter no. of parents and children',min_value=0.0,max_value=10.0,step=1.0)
	FARE = st.sidebar.number_input("Enter Fare amount",min_value=0.0,max_value=10000.0,step=1.0)
	EMBARKED = st.sidebar.selectbox("Embarked(0 - C, 1 - S, 2 - Q)",('0','1','2'))
	data = {'Pclass':PCLASS,'Sex':SEX,'Age':AGE,'SibSp':SIBSP,'Parch':PARCH, 'Fare':FARE, 'Embarked':EMBARKED}
	features = pd.DataFrame(data,index=[0])
	return features

data = user_input_features()
st.subheader("User Input Parameters")
st.write(data)

Titanic_data = pd.read_csv("Titanic_train.csv")
Titanic_data.drop(columns = ['PassengerId','Ticket','Name','Cabin'],inplace=True)
Titanic_data['Embarked'] = Titanic_data['Embarked'].fillna('S')
Titanic_data['Age'] = Titanic_data['Age'].fillna(Titanic_data['Age'].mean())

encoder = LabelEncoder()
Titanic_data['Sex'] = encoder.fit_transform(Titanic_data['Sex'])
Titanic_data['Embarked'] = encoder.fit_transform(Titanic_data['Embarked'])

X = Titanic_data.iloc[:,1:]
Y = Titanic_data.iloc[:,0]
model = LogisticRegression()
model.fit(X,Y)


prediction = model.predict(X)
prediction_probability = model.predict_proba(data)

st.subheader('Predicted Result')
st.write('The person may have been Survived' if prediction_probability[0][1]>0.5 else 'The person may not have been Survived')

st.subheader('Prediction Probability')
st.write(prediction_probability)
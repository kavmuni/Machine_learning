import streamlit as st
import pandas as pd
import joblib as jb

st.title("HR job switch application")

train_df = pd.read_csv('train.csv')


# inputs
city = st.selectbox('City', train_df['city'].unique())
city_development_index = st.number_input('City Development Index', min_value=0.0, max_value=1.0, step=0.01)
gender = st.selectbox('Gender', train_df['gender'].unique())
relevent_experience = st.selectbox('Relevent Experience', train_df['relevent_experience'].unique())
enrolled_university = st.selectbox('Enrolled University', train_df['enrolled_university'].unique())
education_level     = st.selectbox('Education Level', train_df['education_level'].unique())
major_discipline    = st.selectbox('Major Discipline', train_df['major_discipline'].unique())
experience          = st.selectbox('Experience', train_df['experience'].unique())
company_size        = st.selectbox('Company Size', train_df['company_size'].unique())
company_type        = st.selectbox('Company Type', train_df['company_type'].unique())
last_new_job        = st.selectbox('Last New Job', train_df['last_new_job'].unique())
training_hours      = st.number_input('Training Hours', min_value=0, max_value=1000, step=1)

# load the model
model = jb.load('best_model_HR_analysis.pkl')

# form all inouts as a dataframe to be passed into model
input_df = pd.DataFrame({
    'city': [city],
    'city_development_index': [city_development_index],
    'gender': [gender],
    'relevent_experience': [relevent_experience],
    'enrolled_university': [enrolled_university],
    'education_level': [education_level],
    'major_discipline': [major_discipline],
    'experience': [experience],
    'company_size': [company_size],
    'company_type': [company_type],
    'last_new_job': [last_new_job],
    'training_hours': [training_hours]
})

#predict button
if st.button('Predict'):
  # Output
  target = model.predict(input_df)
  st.write(f'Target: {target[0]}')

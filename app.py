import streamlit as st
import pickle
import pandas as pd
import shap
from streamlit_shap import st_shap
import numpy as np
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# Airbnb Rentals\' Price Prediction App
for New Brunswick Area!
""")
st.write('---')
st.write('This is a simple exersice of\
    creating a machine learning based App with Streamlit Library.\
    \nWhere we try to predict the price of Airbnb listings per night based\
    on a reduced number of numerical features and evaluating its\
    corresponding impact on the predicition by ploting SHAP values.')


# Sidebar
# Header of Specify Input Parameters
st.title('Choose you stays\' features:')

def user_input_features():

    accommodates = st.slider('Number of guests', 
                            1, 16, 5, 1)
    maximum_nights = st.slider('Duration of the stays (in days)', 
                              1, 1125, 3, 1)
    number_of_reviews = st.slider('Number of reviews', 
                              1, 582, 36, 10)
    reviews_per_month = st.slider('Average reviews per month', 
                              1, 14, 7, 1)
    st.header('Customers\' appreciations of the rentals on a scale of 5')
    review_scores_rating = st.selectbox('Global Rating',
                               [1,2,3,4,5], 3)
    review_scores_accuracy = st.selectbox('Accuracy',
                               [0,1,2,3,4,5], 3)
    review_scores_cleanliness = st.selectbox('Cleanliness',
                               [0,1,2,3,4,5],3)
    review_scores_checkin = st.selectbox('Checkin',
                               [0,1,2,3,4,5],3)
    review_scores_communication = st.selectbox('Communication',
                               [0,1,2,3,4,5],3)
    review_scores_location = st.selectbox('Location',
                               [0,1,2,3,4,5],3)
    review_scores_value = st.selectbox('Value',
                                        [0,1,2,3,4,5],3)
    data = {
            'accommodates': accommodates,
            'maximum_nights': maximum_nights, 
            'number_of_reviews': number_of_reviews,
            'reviews_per_month': reviews_per_month,
            'review_scores_rating': review_scores_rating,
            'review_scores_accuracy': review_scores_accuracy,
            'review_scores_cleanliness': review_scores_cleanliness,
            'review_scores_checkin': review_scores_checkin,
            'review_scores_communication': review_scores_communication,
            'review_scores_location': review_scores_location,
            'review_scores_value': review_scores_value,
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Load Model from github repo
save ='/app/airbnb_rentals_ml_app/rfr.pkl' #githublink
model_saved = pickle.load(open(save, 'rb'))
# Apply Model to Make Prediction
prediction = model_saved.predict(df)

st.header('Predicted Price in USD Per Night of Stay ')

# Computing SHAP values
explainer = shap.TreeExplainer(model_saved)
shap_values = explainer.shap_values(df)

# plot the predicted price with var impact
st_shap(shap.force_plot(explainer.expected_value, shap_values, df))

#Plot detailed SHAP impact of the predicted price
st.header('Impact by SHAP values')

st.pyplot(shap.summary_plot(shap_values, df),bbox_inches='tight',dpi=500,pad_inches=0)

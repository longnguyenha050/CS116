import pickle
import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
from utilities import numerical_rating
import numpy as np

# Load the model
filename = "saved_models/xgboost_model.pkl"
rf_model = pickle.load(open(filename, "rb"))

st.set_page_config(page_title="Real Estate Price Prediction", layout="wide")


def select_widget(key):
    return st.selectbox(
        key=key,
        label="",
        label_visibility="collapsed",
        options=(
            "Very Poor", "Poor", "Fair", "Below Average", "Average","Cannot Say", "Above Average", "Good", "Very Good", "Excellent", "Very Excellent"
        )
    )


def tech():
    pass


def pred():
    # st.markdown(
    #     "<p style='text-align: justify;'></p>",
    #     unsafe_allow_html=True)

    st.title("RETAIL HOUSE PRICE ESTIMATOR")

    option_overallQuality = select_widget("overallQuality")
    option_overallQuality = numerical_rating(option_overallQuality)
    st.write("Rates build material quality and house finish")

    option_overallCondition = select_widget("overallCondition")
    option_overallCondition = numerical_rating(option_overallCondition)
    st.write("Rates overall house condition")

    livingRoomArea = st.number_input("", min_value=334.0, max_value=4650.0)
    st.write("Living Room Area (in square feet)")

    basementArea = st.number_input("", min_value=0.0, max_value=6100.0)
    st.write("Basement Area (in square feet)")

    firstFloorArea = st.number_input("", min_value=334.0, max_value=4690.0)
    st.write("Total First Floor Area (in square feet)")

    type1FinishedArea = st.number_input("", min_value=0.0, max_value=5660.0)
    st.write("First floor area (in square feet) already constructed")

    secondFloorArea = st.number_input("", min_value=0.0, max_value=2860.0)
    st.write("Total Second Floor Area (in square feet)")

    lotArea = st.number_input("", min_value=1300.0, max_value=21500.0)
    st.write("Total Lot Area (in square feet)")

    yearBuilt = st.number_input("", min_value=1872, max_value=2010)
    st.write("Original Construction Year")

    bathAboveGrade = st.number_input("", min_value=0, max_value=3)
    st.write("No of Bathrooms above Grade")

    yearGarageBuilt = st.number_input("", min_value=1895, max_value=2010)
    st.write("Original Garage Construction Year")

    porchArea = st.number_input("", min_value=0.0, max_value=742.0)
    st.write("Porch Area (in square feet)")

    garageArea = st.number_input("", min_value=0.0, max_value=1448.0)
    st.write("Garage Area (in square foot)")

    garageCarCapacity = st.number_input("", min_value=0, max_value=5)
    st.write("Garage Car Capacity")

    if st.button("Submit"):
        features = np.array([[
            option_overallQuality, option_overallCondition, livingRoomArea, basementArea,
            firstFloorArea, type1FinishedArea, secondFloorArea, lotArea, yearBuilt,
            bathAboveGrade, yearGarageBuilt, porchArea, garageArea, garageCarCapacity
        ]])
        preds = rf_model.predict(features)  # Kết quả là một numpy array
        preds_final = round(preds[0], 2)    # ✅ Lấy phần tử đầu và làm tròn
        st.info(f'**The overall selling price of the house is {int(round(preds_final, 2))}**$')


with st.sidebar:
    choose = option_menu("Welcome", ["Home", "Predictor", "ML Code"],
                         icons=['house', 'stack', 'cpu', 'terminal'],
                         menu_icon='building', default_index=0,
                         styles={
                             "container": {"padding": "5!important", "background-color": "#1a1a1a"},
                             "icon": {"color": "White", "font-size": "25px"},
                             "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px",
                                          "--hover-color": "#4d4d4d"},
                             "nav-link-selected": {"background-color": "#4d4d4d"},
                         })


def ml():
    st.write(
        "To view the complete code for the end-to-end project, visit our [GitHub](https://github.com/sinhasagar507/self-curated-learning/tree/master/Statistical%20ML/Housing%20Prices%20Prediction)")
    components.iframe(
        "https://www.kaggle.com/embed/sagarsinha/housing-prices-prediction?kernelSessionId=112546641",
        height=1000, )


if choose == "Home":
    st.title('Machine Learning-Real Estate Sector')
    st.subheader("Use Case")
    st.markdown(
        "<p style='text-align: justify;'>The purpose of house price prediction is to provide a basis for pricing between buyers and sellers. By viewing transaction records, buyers can understand whether they have received a fair price for a house, and sellers can evaluate the price at which they can sell a house along a specific road section</p>"
        , unsafe_allow_html=True)
    st.write('')
    st.subheader("Solution?")
    st.markdown(
        '''
            <p style='text-align: justify;'>This mini-project is an attempt at understanding various residential properties which predominantly affects house prices. A description of the most relevant parameters affecting sale prices. The sale prices, which are often a result of several parameters, can aid in 
            <br>
            <p>Note, all areas are in <b>SQUARE FEET</b></p>
            <ul>
                <li>Overall Build Material Quality - Scored on a scale from 1 to 10, with 1 representing "Very Poor" and 10 representing "Excellent"</li>
                <li>Overall House Condition - Scored on a scale from 1 to 10, with a similar description as in Overall Build Material Quality</li>
                <li>Living Room Area</li>
                <li>First Floor Finished Area</li>
                <li>Basement Area</li>
                <li>First Floor Area</li>
                <li>Second Floor Area</li>
                <li>Lot Area</li>
                <li>Year of Initial Construction</li>
                <li>Bathroom Rating</li>
                <li>Year when garage was built</li>
                <li>Porch Area</li>
                <li>Garage Area</li>
                <li>Garage Car Capacity - Based on average Car Size in the given area </li>
            </ul>
            </p>
        '''
        ,
        unsafe_allow_html=True)

# elif choose == "Tech Stack":
#     tech()
elif choose == "Predictor":
    pred()
elif choose == "ML Code":
    ml()

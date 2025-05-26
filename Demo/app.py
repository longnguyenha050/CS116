import pickle
import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the model (đảm bảo đường dẫn đúng)
filename = "/content/ML/saved_models/xgboost_model.pkl"
try:
    rf_model = pickle.load(open(filename, "rb"))
except FileNotFoundError:
    st.error(f"Error: Model file not found at {filename}")
    rf_model = None

st.set_page_config(page_title="Real Estate Price Prediction", layout="wide")

def numerical_rating1(userRating1):
    if userRating1 == "Very Poor":
        return 1
    elif userRating1 == "Poor":
        return 2
    elif userRating1 == "Fair":
        return 3
    elif userRating1 == "Below Average":
        return 4
    elif userRating1 == "Average" or userRating1 == "Cannot Say":
        return 5
    elif userRating1 == "Above Average":
        return 6
    elif userRating1 == "Good":
        return 7
    elif userRating1 == "Very Good":
        return 8
    elif userRating1 == "Excellent":
        return 9
    elif userRating1 == "Very Excellent":
        return 10
def numerical_rating(userRating2):
    if userRating2 == "NA":
        return 0
    elif userRating2 == "Poor" or userRating2 == "Unf":
        return 1
    elif userRating2 == "Fair" or userRating2 == "RFn":
        return 2
    elif userRating2 == "Average" or userRating2 == "Typical" or userRating2 == "Fin":
        return 3
    elif userRating2 == "Good":
        return 4
    elif userRating2 == "Excellent":
        return 5
    else:
        return 0 # Giá trị mặc định nếu không khớp

def category(userChosing):
  if userChosing == "Commercial":
    return "C (all)"
  elif userChosing == "Floating Village Residential":
    return "FV"
  elif userChosing == "Residential High Density":
    return "RH"
  elif userChosing == "Residential Low Density":
    return "RL"
  elif userChosing == "Residential Medium Density":
    return "RM"
  else:
    return 0


CentralAir = ['Yes','No']

from sklearn.preprocessing import LabelEncoder

def create_label_encoder(label_list):
    """
    Khởi tạo và fit LabelEncoder từ danh sách nhãn ban đầu.
    """
    le = LabelEncoder()
    le.fit(label_list)
    return le

def encode_labels(le, labels):
    """
    Mã hóa 1 chuỗi hoặc danh sách chuỗi bằng encoder đã fit.
    """
    if isinstance(labels, str):
        labels = [labels]
    return le.transform(labels)


def select_widget(key):
    """
    Hàm tạo Streamlit selectbox với các tùy chọn khác nhau dựa trên key.
    """
    if key == "CentralAir":
        options_list = ("Yes", "No")
    elif key in ["OverallQual"]:
         options_list = ("Very Poor", "Poor", "Fair", "Below Average", "Average","Cannot Say", "Above Average", "Good", "Very Good", "Excellent", "Very Excellent")
    elif key in ["KitchenQual",  "ExterQual", "GarageCond", "FireplaceQu"]:
        options_list = ("NA", "Poor", "Fair", "Average", "Good", "Excellent")
    elif key == "MSZoning":
        # Cần thay thế bằng danh sách các giá trị duy nhất thực tế của MSZoning từ dữ liệu huấn luyện
        options_list = ["Commercial", "Floating Village Residential", "Residential High Density", "Residential Low Density", "Residential Medium Density"] # Thay thế bằng danh sách của bạn
    elif key == "Neighborhood":
        # Cần thay thế bằng danh sách các giá trị duy nhất thực tế của Neighborhood từ dữ liệu huấn luyện
        options_list = ["Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr", "Crawfor", "Edwards", "Gilbert	", "IDOTRRr", "MeadowV", "Mitchel", "NAmes", "NoRidge", "NPkVill", "NridgHt", "NWAmes", "OldTown", "SWISU"] # Thay thế bằng danh sách của bạn
    elif key == "GarageType":
         # Cần thay thế bằng danh sách các giá trị duy nhất thực tế của GarageType từ dữ liệu huấn luyện
         options_list = ["2Types", "Attchd", "Basment", "BuiltIn", "CarPort", "Detchd","NA"] # Thay thế bằng danh sách của bạn
    elif key == "GarageFinish":
         # Cần thay thế bằng danh sách các giá trị duy nhất thực tế của GarageFinish từ dữ liệu huấn luyện
         options_list = ["Fin", "Unf", "RFn", "None"] # Thay thế bằng danh sách của bạn

    else:
        st.error(f"Unknown key for select_widget: {key}")
        return None

    return st.selectbox(
        key=key,
        label="",
        label_visibility="collapsed",
        options=options_list
    )

def tech():
    pass

def pred():
    st.title("RETAIL HOUSE PRICE ESTIMATOR")

    option_overallQuality = select_widget("OverallQual") 
    overallQuality_numeric = numerical_rating1(option_overallQuality) # ordinal
    st.write("Rates build material quality and house finish")

    option_centralAir = select_widget("CentralAir")
    le = create_label_encoder(CentralAir)
    centralAir_numeric = encode_labels(le, option_centralAir) # label encode
    st.write("Does the house have Central Air?")

    option_kitchenQuality = select_widget("KitchenQual")
    kitchenQuality_numeric = numerical_rating(option_kitchenQuality) # ordinal
    st.write("Rates quality and kitchen finish")

    option_ExterQuality = select_widget("ExterQual")
    exterQuality_numeric = numerical_rating(option_ExterQuality) # ordinal
    st.write("Rates Exterior material quality")

    GarageCars = st.number_input("",min_value=0, max_value=5) # Giả định max 5 xe
    st.write("Garage Car Capacity")

    option_overallGarageCond = select_widget("GarageCond")
    overallGarageCond_numeric = numerical_rating(option_overallGarageCond) # ordinal
    st.write("Rates Garage quality")

    option_mszoning = select_widget("MSZoning")
    option_mszoning = category(option_mszoning)
    if option_mszoning is not None:
        st.write("General zoning classification")

    option_FireplaceQu = select_widget("FireplaceQu")
    fireplaceQu_numeric = numerical_rating(option_FireplaceQu) # ordinal
    st.write("Rates Fireplace quality")

    FullBath = st.number_input("",min_value=0, max_value=4) # Giả định max 3 phòng
    st.write("Number of full bathrooms above grade")

    option_GarageFinish = select_widget("GarageFinish")
    garageFinish_numeric = numerical_rating(option_GarageFinish) # ordinal
    st.write("Interior finish of the garage")

    GrLivArea = st.number_input("",min_value=334.0, max_value=4000.0)
    st.write("Total Ground Living Area (in square feet)")

    option_neighborhood = select_widget("Neighborhood")
    if option_neighborhood is not None:
        st.write("Physical locations within Ames city limits")

    TotalBsmtSF = st.number_input("",min_value=0.0, max_value=5660.0)
    st.write("Total square feet of basement area")

    option_GarageType = select_widget("GarageType")
    if option_GarageType is not None:
        st.write("Type of garage")


    # -- Thêm các selectbox cho các biến danh nghĩa mới --




    from scipy.special import boxcox1p

    neighborhood_values = [
        'Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr',
        'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel',
        'NAmes', 'NPkVill','NWAmes','NoRidge', 'NridgHt', 'OldTown',
        'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker'
    ]

    mszoning_values = ['C (all)', 'FV', 'RH', 'RL', 'RM']
    garage_type_values = [ '2Types', 'Attchd', 'Basment',  'BuiltIn', 'CarPort',  'Detchd', 'NA']


    import pandas as pd

    def get_dummies_fixed(input_dict, dummy_columns_dict):
        df = pd.DataFrame([input_dict])

        for col, categories in dummy_columns_dict.items():
            for cat in categories:
                df[f"{col}_{cat}"] = 1 if df[col][0] == cat else 0
            df.drop(columns=[col], inplace=True)  # Xoá cột gốc

        return df



    skewed_features = [
        "GrLivArea", "ExterQual", "TotalBsmtSF", "KitchenQual", "OverallQual",
        "FireplaceQu", "FullBath", "GarageFinish", "GarageCars", "GarageCond", #"CentralAir"
    ]
    lam = 0.15



    # -- Nút Submit và Dự đoán --
    if st.button("Submit"):
        if rf_model is not None:
            input_data = {
                'Neighborhood': option_neighborhood,
                'MSZoning': option_mszoning,
                'GarageType': option_GarageType,
            }

            dummy_dict = {
                'Neighborhood': neighborhood_values,
                'MSZoning': mszoning_values,
                'GarageType': garage_type_values
            }

            one_hot_df = get_dummies_fixed(input_data, dummy_dict)

            numeric_data = {
                'OverallQual': overallQuality_numeric,
                'CentralAir': centralAir_numeric,
                'KitchenQual': kitchenQuality_numeric,
                'ExterQual': exterQuality_numeric,
                'GarageCars': GarageCars,
                'GarageCond': overallGarageCond_numeric,
                'FireplaceQu': fireplaceQu_numeric,
                'FullBath': FullBath,
                'GarageFinish': garageFinish_numeric,
                'GrLivArea': GrLivArea,
                'TotalBsmtSF': TotalBsmtSF,
            }

            numerical_df = pd.DataFrame([numeric_data])
            final_input_df = numerical_df.join(one_hot_df)

            # ✅ Áp dụng Box-Cox transform với lambda cho các cột bị skew
            for col in skewed_features:
                if col in final_input_df:
                    final_input_df[col] = boxcox1p(final_input_df[col], lam)

            try:
                features = final_input_df.to_numpy()
                preds = rf_model.predict(features)
                # Giải log
                preds_final = np.expm1(preds[0])
                st.info(f'**The overall selling price of the house is {int(round(preds_final, 2))}**$')

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.warning("Please ensure that all inputs are provided and the data is preprocessed correctly.")
        else:
            st.warning("Model not loaded. Please check the model file path.")


# -- Phần còn lại của code (sidebar, home, ml) giữ nguyên --
with st.sidebar:
    choose = option_menu("Welcome", ["Home", "Predictor"],
                         icons=['house', 'stack','terminal'],
                         menu_icon='building', default_index=0,
                         styles={
                             "container": {"padding": "5!important", "background-color": "#1a1a1a"},
                             "icon": {"color": "White", "font-size": "25px"},
                             "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px",
                                          "--hover-color": "#4d4d4d"},
                             "nav-link-selected": {"background-color": "#4d4d4d"},
                         })


if choose == "Home":
    st.title('Machine Learning-Real Estate Sector')
    st.subheader("Use Case")
    st.markdown(
        "<p style='text-align: justify;'>The purpose of house price prediction is to provide a basis for pricing between buyers and sellers. By viewing transaction records, buyers can understand whether they have received a fair price for a house, and sellers can evaluate the price at which they can sell a house along a specific road section</p>"
        , unsafe_allow_html=True)
    st.write('')
    st.subheader("Features")
    st.markdown(
        '''
            <p>Note, all areas are in <b>SQUARE FEET</b></p>
            <ul>
              <li>Overall Build Material Quality</li>
              <li>Central Air Conditioning</li>
              <li>Kitchen Quality</li>
              <li>Exterior Wall Quality</li>
              <li>Garage Car Capacity</li>
              <li>Garage Condition</li>
              <li>Zoning Classification (MSZoning)</li>
              <li>Fireplace Quality</li>
              <li>Number of Full Bathrooms</li>
              <li>Garage Finish Level</li>
              <li>Above Ground Living Area (GrLivArea)</li>
              <li>Neighborhood</li>
              <li>Total Basement Area</li>
              <li>Garage Type</li>
            </ul>
        '''
        ,
        unsafe_allow_html=True)

elif choose == "Predictor":
    pred()
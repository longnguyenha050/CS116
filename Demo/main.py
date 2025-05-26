from fastapi import FastAPI
from parameters.body_params import HouseFeatures
from xgboost import Booster, DMatrix
import numpy as np

# Load model đã huấn luyện
booster_model = Booster()
booster_model.load_model("saved_models/xgboost_model.json")

# Khởi tạo FastAPI app
app = FastAPI()

@app.post("/get-House-Fs")
def get_House_fs(request: HouseFeatures):
    def map_quality(value: str, default=5):
        return {
            "Very Poor": 1, "Poor": 2, "Fair": 3, "Below Average": 4,
            "Average": 5, "Above Average": 6, "Good": 7,
            "Very Good": 8, "Excellent": 9, "Very Excellent": 10,
            "Cannot Say": 5
        }.get(value, default)

    def round_or_default(val, default):
        return int(round(val, 2)) if val is not None else default

    # Tiền xử lý đầu vào
    overallQuality = map_quality(request.overallQuality)
    livingRoomArea = round_or_default(request.livingRoomArea, 334)
    basementArea = round_or_default(request.basementArea, 0)
    firstFloorArea = round_or_default(request.firstFloorArea, 334)
    type1FinishedArea = round_or_default(request.type1FinishedArea, 0)
    secondFloorArea = round_or_default(request.secondFloorArea, 0)
    lotArea = round_or_default(request.lotArea, 1300)
    yearBuilt = round_or_default(request.yearBuit, 1872)
    bathAboveGrade = round_or_default(request.bathAboveGrade, 0)
    yearGarageBuilt = round_or_default(request.yearGarageBuilt, 1895)
    porchArea = round_or_default(request.porchArea, 0)
    garageArea = round_or_default(request.garageArea, 0)
    garageCarCapacity = round_or_default(request.garageCarCapacity, 0)
    overallCondition = map_quality(request.overallCondition, 1)

    # Dự đoán với DMatrix
    input_features = np.array([[
        overallQuality, livingRoomArea, basementArea, firstFloorArea, type1FinishedArea,
        secondFloorArea, lotArea, yearBuilt, bathAboveGrade, yearGarageBuilt,
        porchArea, garageArea, garageCarCapacity, overallCondition
    ]])
    dmatrix = DMatrix(input_features)
    prediction = booster_model.predict(dmatrix)
    predicted_price = round(float(prediction[0]), 2)

    return {"Predicted House Price": predicted_price}

from pydantic import BaseModel, Field
from typing import Union


class HouseFeatures(BaseModel):
    overallQuality: Union[str, None] = Field(None, example="Excellent")
    livingRoomArea: Union[str, None] = Field(None, example="1800", description="Measured in square feet. Enter a value between 334 and 5462 inclusive")
    basementArea: Union[str, None] = Field(None, example="2000", description="Measured in square feet. Enter a value between 0 and 6110 inclusive")
    firstFloorArea: Union[str, None] = Field(None, example="4000", description="Measured in square feet. Enter a value between 334 and 4692 inclusive")
    type1FinishedArea: Union[str, None] = Field(None, example="5000", description="Measured in square feet. Enter a value between 0 and 5664 inclusive")
    secondFloorArea: Union[str, None] = Field(None, example="2000", description="Measured in square feet. Enter a value between 0 and 2065 inclusive")
    lotArea: Union[str, None] = Field(None, example="3000", description="Measured in square feet. Enter a value between 1300 and 21500 inclusive")
    yearBuilt: Union[str, None] = Field(None, example="2002", description="Year range between 1872 and 2010 inclusive")
    bathAboveGrade: Union[str, None] = Field(None, example="2", description="Enter a value between 0 and 3 inclusive")
    yearGarageBuilt: Union[str, None] = Field(None, example="2002", description="Year range between 1895 and 2010 inclusive")
    porchArea: Union[str, None] = Field(None, example="500", description="Measured in square feet. Enter a value between 0 and 742 inclusive")
    garageArea: Union[str, None] = Field(None, example="700", description="Measured in square feet. Enter a value between 0 and 1448 inclusive")
    garageCarCapacity: Union[str, None] = Field(None, example="2", description="Enter a value between 0 and 5 inclusive")
    overallCondition: Union[str, None] = Field(None, example="Average")

    class Config:
        join_schema_extra = {
            "example": {
                "overallQuality": "Excellent",
                "livingRoomArea": "1800",
                "basementArea": "2000",
                "firstFloorArea": "8000",
                "type1FinishedArea": "5000",
                "secondFloorArea": "7000",
                "lotArea": "3000",
                "yearBuilt": "2002",
                "bathAboveGrade": "5",
                "yearGarageBuilt": "2002",
                "porchArea": "1000",
                "garageArea": "700",
                "garageCarCapacity": "2",
                "overallCondition": "Average"
            }
        }

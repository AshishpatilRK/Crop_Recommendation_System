import os
import sys
from crop_recommendn_prdn.exceptions.exception import CustomException
from crop_recommendn_prdn.logging.logger import logging
import pandas as pd 
import numpy as np
from dataclasses import dataclass

from crop_recommendn_prdn.utils.util import load_object

class PredictPipline:
    def __init__(self):
        pass


    def predict(self,features):
        try:
            ## Load pickel File
            ## This Code Work in /any system
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
            model_path = os.path.join("artifacts","model.pkl")


            ## Load Pickel File
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)


            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            logging.info("Error Occure in Prediction Pipline")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
            N:int,
            P:int,
            K:int,
            temperature:int,
            humidity:int,
            ph:int,
            rainfall:int
            ):

        self.N = N
        self.P = P
        self.K = K
        self.temperature = temperature
        self.humidity = humidity
        self.ph = ph
        self.rainfall = rainfall
       
        

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "N":[self.N],
                "P":[self.P],  
                "K":[self.K], 
                "temperature":[self.temperature],
                "humidity":[self.humidity],
                "ph":[self.ph],
                "rainfall":[self.rainfall]
            }
            data = pd.DataFrame(custom_data_input_dict)
            logging.info("Data Frame Gathered")
            return data

        except Exception as e:
            logging.info("Error Occured In Predict Pipline")
            raise CustomException(e, sys)

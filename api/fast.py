import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pytz
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}


# @app.get("/predict")
# params = {
#         "pickup_datetime": ['2012-10-06 12:10:20'],
#         "pickup_longitude": [40.7614327],
#         "pickup_latitude": [-73.9798156],
#         "dropoff_longitude": [40.6413111],
#         "dropoff_latitude": [-73.9797156],
#         "passenger_count": [1]
#     }


@app.get("/predict")
def predict(pickup_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count):
    params = {
        "pickup_datetime": [pickup_datetime],
        "pickup_longitude": [pickup_longitude],
        "pickup_latitude": [pickup_latitude],
        "dropoff_longitude": [dropoff_longitude],
        "dropoff_latitude": [dropoff_latitude],
        "passenger_count": [passenger_count]
        }

    # define X_pred as a df
    X_pred = pd.DataFrame.from_dict(params)
    
    # handle the key column
    X_pred["key"]='2021-08-19 14:31:00.000000119'
    first_column = X_pred.pop('key')
    X_pred.insert(0, 'key', first_column)
    
    # localize the datetime provided by the developer
    ## create a datetime object from the user provided datetime
    pickup_datetime = X_pred['pickup_datetime'].values[0]
    pickup_datetime = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")

    ## localize the user datetime with NYC timezone
    eastern = pytz.timezone("US/Eastern")
    localized_pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)

    ## convert localize datetime to UTC
    utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc)

    ## convert object to datetime format
    X_pred['pickup_datetime'] = utc_pickup_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")
    
    pipeline = joblib.load('model.joblib')
    y_pred = pipeline.predict(X_pred)
    
    return {'prediction': y_pred}
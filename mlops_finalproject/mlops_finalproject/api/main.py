import os
import sys
import joblib
import logging
import pandas as pd

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from fastapi import FastAPI
from starlette.responses import JSONResponse
from sklearn.model_selection import train_test_split
from predictor.predict import ModelPredictor
from api.models.models import HotelReservation
from train.train_data import HotelReservationsDataPipeline


logger = logging.getLogger(__name__) # Indicamos que tome el nombre del modulo
logger.setLevel(logging.INFO) # Configuramos el nivel de logging

formatter = logging.Formatter('%(asctime)s:%(name)s:%(module)s:%(levelname)s:%(message)s') # Creamos el formato

file_handler = logging.FileHandler('main_fast_api.log') # Indicamos el nombre del archivo

file_handler.setFormatter(formatter) # Configuramos el formato

logger.addHandler(file_handler) # Agregamos el archivo

DATASETS_DIR = './data/'
URL = '/Users/norma.perez/Documents/GitHub/MLOps_FinalProject/mlops_finalproject/mlops_finalproject/data/Hotel_Reservations.csv'
DROP_COLS = ['Booking_ID']
RETRIEVED_DATA = 'retrieved_data.csv'


SEED_SPLIT = 404
TRAIN_DATA_FILE = DATASETS_DIR + 'train.csv'
TEST_DATA_FILE  = DATASETS_DIR + 'test.csv'


TARGET = 'booking_status'
FEATURES = ['no_of_adults','no_of_children','no_of_weekend_nights','no_of_week_nights','type_of_meal_plan','required_car_parking_space','room_type_reserved','lead_time','arrival_year','arrival_month','arrival_date','market_segment_type','repeated_guest','no_of_previous_cancellations','no_of_previous_bookings_not_canceled','avg_price_per_room','no_of_special_requests','booking_status']
NUMERICAL_VARS = ['no_of_adults','no_of_children','no_of_weekend_nights','no_of_week_nights','required_car_parking_space','lead_time','arrival_year','arrival_month','arrival_date','repeated_guest','no_of_previous_cancellations','no_of_previous_bookings_not_canceled','avg_price_per_room','no_of_special_requests']
CATEGORICAL_VARS = ['type_of_meal_plan','room_type_reserved','market_segment_type']


NUMERICAL_VARS_WITH_NA = []
CATEGORICAL_VARS_WITH_NA = []
NUMERICAL_NA_NOT_ALLOWED = [var for var in NUMERICAL_VARS if var not in NUMERICAL_VARS_WITH_NA]
CATEGORICAL_NA_NOT_ALLOWED = [var for var in CATEGORICAL_VARS if var not in CATEGORICAL_VARS_WITH_NA]


SEED_MODEL = 404
SELECTED_FEATURES = ['lead_time', 'avg_price_per_room', 'no_of_special_requests', 'arrival_date', 'arrival_month', 'no_of_week_nights']

TRAINED_MODEL_DIR = './mlops_finalproject/models/'
MODEL_NAME = 'extra_trees_classifier_model'
PIPELINE_NAME = 'extra_trees_classifier_pipeline'
MODEL_SAVE_FILE = f'{MODEL_NAME}_output.pkl'
PIPELINE_SAVE_FILE = f'{PIPELINE_NAME}_output.pkl'
MODEL_SAVE_PATH = TRAINED_MODEL_DIR + MODEL_SAVE_FILE


app = FastAPI()

@app.get('/', status_code=200)
async def healthcheck():
    logger.info("Healthy was checked.")
    return 'HotelReservation classifier is all ready to go!'

@app.post('/predict')
def predict(hotelreservation_features: HotelReservation):
    predictor = ModelPredictor("/Users/norma.perez/Documents/GitHub/MLOps_FinalProject2/mlops_finalproject/mlops_finalproject/models/extra_trees_classifier_model_output.pkl")
    X = [hotelreservation_features.no_of_week_nights,
        hotelreservation_features.lead_time,
        hotelreservation_features.arrival_month,
        hotelreservation_features.arrival_date,
        hotelreservation_features.avg_price_per_room,
        hotelreservation_features.no_of_special_requests]
    
    logger.info(f"HotelReservations features: {X}")

    prediction = predictor.predict([X])
    logger.info(f"Resultado de predicción: {prediction}")
    return JSONResponse(f"Resultado predicción: {prediction}")

@app.post('/train')
def train_model():
    try:
        # Instantiate the training pipeline
        hotelreservations_data_pipeline = HotelReservationsDataPipeline(
            seed_model=404,
            numerical_vars=['no_of_adults','no_of_children','no_of_weekend_nights','no_of_week_nights','required_car_parking_space','lead_time','arrival_year','arrival_month','arrival_date','repeated_guest','no_of_previous_cancellations','no_of_previous_bookings_not_canceled','avg_price_per_room','no_of_special_requests'],  # Provide your numerical variables here
            categorical_vars_with_na=[],  # Provide categorical variables with missing values here
            numerical_vars_with_na=[],  # Provide numerical variables with missing values here
            categorical_vars=['type_of_meal_plan','room_type_reserved','market_segment_type'],  # Provide your categorical variables here
            selected_features=['lead_time', 'avg_price_per_room', 'no_of_special_requests', 'arrival_date', 'arrival_month', 'no_of_week_nights']  # Provide your selected features here
        )

        # Read data
        df = pd.read_csv(DATASETS_DIR + RETRIEVED_DATA)

        # Load and preprocess data
        X_train, X_test, y_train, y_test = train_test_split(
                                                        df.drop(TARGET, axis=1),
                                                        df[TARGET],
                                                        test_size=0.2,
                                                        random_state=404
                                                   )
        # Fit the model using the pipeline
        extra_trees_classifier_model = hotelreservations_data_pipeline.fit_extra_trees_classifier(X_train, y_train)
        
        result = joblib.dump(extra_trees_classifier_model, MODEL_SAVE_PATH)
        logger.debug(f"ACTION -> Train model saved in {result}")
        return JSONResponse(content="Extra Tree Classifier Model training completed.")
    
    except Exception as e:
        logger.error(f"An error occurred during model training: {str(e)}")
        return JSONResponse(content="Error occurred during model training.", status_code=500)

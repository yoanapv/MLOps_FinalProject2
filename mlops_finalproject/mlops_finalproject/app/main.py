import os
import sys
import logging

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from fastapi import FastAPI
from starlette.responses import JSONResponse

from predictor.predict import ModelPredictor
from api.models.models import HotelReservation
from train.train_data import HotelReservationsDataPipeline


logger = logging.getLogger(__name__) # Indicamos que tome el nombre del modulo
logger.setLevel(logging.INFO) # Configuramos el nivel de logging

formatter = logging.Formatter('%(asctime)s:%(name)s:%(module)s:%(levelname)s:%(message)s') # Creamos el formato

file_handler = logging.FileHandler('main_fast_api.log') # Indicamos el nombre del archivo

file_handler.setFormatter(formatter) # Configuramos el formato

logger.addHandler(file_handler) # Agregamos el archivo


app = FastAPI()

@app.get('/', status_code=200)
async def healthcheck():
    logger.info("Healthy was checked.")
    return 'HotelReservation classifier is all ready to go!'

@app.post('/predict')
def predict(hotelreservation_features: HotelReservation):
    predictor = ModelPredictor("ml_models/extra_trees_classifier_model_output.pkl")
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

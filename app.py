from flask import Flask, request
from loguru import logger
import constants
from model_functions import fit_and_save_model, predict

app = Flask(__name__)
logger.add(sink='logs', colorize=True, format="{time:YYYY-MM-DD HH:mm:ss} [{level}] {message}")
logger.info('API was started.')


@app.route('/health', methods=['GET'])
def check_health():
    logger.info('API service for the model is running and working correctly.')
    return {'API Status': 'API service for the model is running and working correctly.'}


@app.route('/get_predicts', methods=['GET', 'POST'])
def get_predictions():
    data = request.get_json()
    result = predict(data)
    return result


@app.route('/train_model', methods=['GET'])
def start_train_models():
    logger.info(f"The model's training process was started.")
    result = fit_and_save_model()
    logger.info(result)
    return result


if __name__ == '__main__':
    app.run('0.0.0.0', constants.PROD_HOST, debug=True)

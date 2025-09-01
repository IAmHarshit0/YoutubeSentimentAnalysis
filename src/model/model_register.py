import json
import mlflow
import logging 
import os

mlflow.set_tracking_uri("http://ec2-13-61-146-35.eu-north-1.compute.amazonaws.com:5000/")

logger = logging.getLogger('model_registeration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('ERROR')

file_handler = logging.FileHandler('model_registeration.py')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except Exception as e:
        logger.error('Unexpected error occured while loading the model info %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        model_version = mlflow.register_model(model_uri, model_name)

        client = mlflow.tracking(model_uri, model_name)
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage='Staging'
        )

        logger.debug(f"Model {model_name} version {model_version.version} registered and transitioned to Staging")
    except Exception as e:
        logger.error('Error during model registeration: %s', e)
        raise

def main():
    try:
        model_info_path = 'experiment_info.json'
        model_info = load_model_info(model_info_path)

        model_name = "yt_chrome_plugin_model"

        register_model(model_name, model_info)
    except Exception as e:
        logger.error('Failed to complete the model registeration process: %s', e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
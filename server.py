from flask import Flask, Response, request
import cloudpickle
import tensorflow as tf
import mlflow.pyfunc
import pandas as pd
import pythonmodel


app = Flask(__name__)

mlflow_pyfunc_model_path = "model_path"
loaded_model = None


@app.route('/predict', methods=["POST"])
def predict():

	error = ''

	if request.method == "POST":
		data_dict = pd.DataFrame.from_dict(request.json)
		test_predictions = loaded_model.predict(data_dict)
		return test_predictions


if __name__ == "__main__":
	loaded_model = mlflow.pyfunc.load_model(mlflow_pyfunc_model_path)
	app.run(host='0.0.0.0', port=5001, debug=True)
	
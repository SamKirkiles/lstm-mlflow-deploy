from flask import Flask, Response, request
import cloudpickle
import tensorflow as tf
import mlflow.pyfunc
import pandas as pd
import pythonmodel


application = Flask(__name__,static_url_path="",static_folder="static")

mlflow_pyfunc_model_path = "model_path"
loaded_model = None

@application.route('/', methods=["GET"])
def root():
	return application.send_static_file('index.html')


@application.route('/predict', methods=["POST"])
def predict():

	error = ''
	if request.method == "POST":
		try:
			data_dict = pd.DataFrame.from_dict(request.json)
			test_predictions = loaded_model.predict(data_dict)
			return test_predictions
		except:
			return "Bad request"

if __name__ == "__main__":
	#loaded_model = mlflow.pyfunc.load_model(mlflow_pyfunc_model_path)
	application.run(debug=True)
	

import cloudpickle
import pythonmodel
import tensorflow as tf
import mlflow.pyfunc 
import pandas as pd

artifacts = {}

conda_env = {
    'channels': ['defaults'],
    'dependencies': [
      'tensorflow={}'.format(tf.__version__),
      'cloudpickle={}'.format(cloudpickle.__version__),
      'pandas={}'.format(pd.__version__),
    ],
    'name': 'lstm_env'
}

mlflow_pyfunc_model_path = "./model_path"
mlflow.pyfunc.save_model(
    path=mlflow_pyfunc_model_path, python_model=pythonmodel.lstmModel(), artifacts=artifacts,
    conda_env=conda_env)\



loaded_model = mlflow.pyfunc.load_model(mlflow_pyfunc_model_path)
data_dict = pd.DataFrame.from_dict({'output_length': [100]})
print(data_dict["output_length"][0])
test_predictions = loaded_model.predict(data_dict)
print(test_predictions)

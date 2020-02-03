import cloudpickle
import lstm_pyfunc_model
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
    path=mlflow_pyfunc_model_path,
    python_model=lstm_pyfunc_model.LstmModel(),
    artifacts=artifacts,
    conda_env=conda_env,
)

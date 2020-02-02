import mlflow.pyfunc
import lstm
import data
import pandas as pd
import pickle


class lstmModel(mlflow.pyfunc.PythonModel):

	def load_context(self, context):

		light_device = "/cpu:0"
		heavy_device = "/cpu:0"

		with open('./saves/state.pkl', 'rb') as f:
			_, _, self.char2ix, self.ix2char = pickle.load(f)

		self.solver = lstm.LSTM(
			num_classes=len(self.char2ix),
			heavy_device=heavy_device,
			light_device=light_device,
			restore=True
		)

	def predict(self, context, model_input):
		dict_input = model_input.to_dict()
		return self.solver.generate(self.char2ix, self.ix2char, dict_input['output_length'][0])

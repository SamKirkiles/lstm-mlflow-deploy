
import tensorflow as tf
import numpy as np
import lstm
import sys
import mlflow
import data
import pickle
import calendar
import time

light_device = "/cpu:0"
heavy_device = "/cpu:0"
test = False
restore = False


def main():

	run_id = np.random.randint(1000)

	if restore:
		with open('./saves/state.pkl', 'rb') as f:
			X, Y, char2ix, ix2char = pickle.load(f)

	else:
		X, Y, char2ix, ix2char = data.read_data("warandpeace.txt", sequence_length=100)

		with open('./saves/state.pkl', 'wb') as f:
			pickle.dump([X, Y, char2ix, ix2char], f)

	train_set = data.train_set(X,Y,128)

	solver = lstm.LSTM(
		num_classes=len(char2ix), 
		heavy_device=heavy_device, 
		light_device=light_device,
		restore=restore
	)

	if test == False:
		solver.train(train_set)
	else:
		print(solver.generate(char2ix, ix2char, 100))

if __name__ == "__main__":
	for o in sys.argv[1:]:
		if o == '--gpu':
			heavy_device = "/gpu:0"
			light_device = "/cpu:0"
		elif o == '--cpu':
			heavy_device = "/cpu:0"
			light_device = "/cpu:0"
		elif o == '--test':
			test = True
		elif o == '--train':
			test = False
		elif o == '--restore':
			restore = True
		else:
			raise ValueError("Unkown argument: " + o)

	main()

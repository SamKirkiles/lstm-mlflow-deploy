
import tensorflow as tf
import numpy as np
import lstm
import sys
import data

gpu = False
test = False
restore = False

def main():
	run_id = np.random.randint(1000)

	
	X, Y, char2ix, ix2char = data.read_data("warandpeace.txt",sequence_length=100)
	# Generator to get random samples
	train_set = data.train_set(X,Y,128)

	solver = lstm.LSTM(num_classes=len(char2ix))


	if test == False:
		solver.train(train_set,restore=restore)
	else:
		print(solver.generate(char2ix,ix2char,1000,restore=restore))

if __name__ == "__main__":
	for o in sys.argv[1:]:
		if o == '--gpu':
			gpu = True
		elif o == '--cpu':
			gpu = False
		elif o == '--test':
			test = True
		elif o == '--train':
			test = False
		elif o == '--restore':
			restore = True

		else:
			raise ValueError("Unkown argument: " + o)

	main()

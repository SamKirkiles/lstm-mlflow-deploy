
import tensorflow as tf
import numpy as np
import lstm
import sys

gpu = False
test = False

def main():
	run_id = np.random.randint(1000)
	data = open('warandpeace.txt','r',encoding='utf-8').read()


	solver = lstm.LSTM()

	if test == False:
		solver.train(data,gpu)


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
		else:
			raise ValueError("Unkown argument: " + o)

	main()

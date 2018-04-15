
import tensorflow as tf
import numpy as np
import lstm
import sys

gpu = False
test = False
restore = False

def main():
	run_id = np.random.randint(1000)
	data = open('warandpeace.txt','r',encoding='utf-8').read()

	solver = lstm.LSTM()

	if test == False:
		solver.train(data,gpu,restore=restore)
	else:
		print(solver.test(data,gpu=gpu,restore=restore))

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

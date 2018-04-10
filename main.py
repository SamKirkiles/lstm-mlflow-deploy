
import tensorflow as tf
import numpy as np
import lstm


def main():
	run_id = np.random.randint(1000)
	data = open('warandpeace.txt','r',encoding='utf-8').read()


	solver = lstm.LSTM()
	solver.train(data)


if __name__ == "__main__":

	main()

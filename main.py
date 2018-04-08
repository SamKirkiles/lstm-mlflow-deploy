
import tensorflow as tf
import numpy as np
import lstm


def main():
	run_id = np.random.randint(1000)
	data = open('input.txt','r',encoding='utf-8').read()

	chars = list(set(data))
	data_size,vocab_size = len(data),len(chars)

	char_to_ix = {ch:i for i,ch in enumerate(chars)}
	ix_to_char = {i:ch for i,ch in enumerate(chars)}

	tables = {'char_to_ix':char_to_ix,'ix_to_char':ix_to_char}

	solver = lstm.LSTM(vocab_size,tables)
	solver.train(data)


if __name__ == "__main__":

	main()

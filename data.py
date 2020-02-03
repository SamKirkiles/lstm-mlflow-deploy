
import numpy as np
import random

def read_data(filename, sequence_length):
	# Read in data from text file
	data = open(filename,'r',encoding='utf-8').read()

	chars = list(set(data))
	data_length = len(data)

	num_samples = data_length//sequence_length

	# Create character lookup dictionaries 
	char2ix = {ch:i for i,ch in enumerate(chars)}
	ix2char = {i:ch for i,ch in enumerate(chars)}

	# Create our batches containing samples of our size sequence length
	X = np.zeros([num_samples,sequence_length])
	Y = np.zeros([num_samples,sequence_length])

	for i in range(num_samples):
		X[i] = np.array([ char2ix[ch] for ch in data[i*sequence_length:(i+1)*sequence_length]])
		Y[i] = np.array([ char2ix[ch] for ch in data[i*sequence_length + 1:(i+1)*sequence_length + 1]])

	return X, Y, char2ix, ix2char

def train_set(X, Y, batch_size):
	while True:
		sample = random.sample(list(np.arange(len(X))), batch_size)
		yield X[sample], Y[sample]

import tensorflow as tf
import numpy as np
import os 
import pickle

class LSTM:

	def __init__(self,num_classes,state_size=512,layers=3,heavy_device=None,light_device=None):
		# Initializes the lstm and builds the graph when run

		self.state_size = state_size
		self.layers = layers
		self.num_classes = num_classes
		self.heavy_device = heavy_device
		self.light_device = light_device

		def __graph__():

			# Build the graph that will process one batch 

			def step(prev,x):

				with tf.device(self.heavy_device):

					# x will be a tensor of shape [batch_size,state_size]
					# prev will be a tensor of shape [2, num_layers, batch_size, state_size]
					# We will unstack this and return a tensor of the same shape to be passed into the next timestep
					# Using embeddings we can reshape our number of features into the size of our desired hidden shape

					# Get weights or initialize if not already in graph
					W = tf.get_variable(name="W",shape=[self.layers, 4, self.state_size, self.state_size],initializer=tf.random_uniform_initializer(minval=0.08,maxval=0.08))
					U = tf.get_variable(name = "U",shape=[self.layers, 4, self.state_size, self.state_size],initializer=tf.random_uniform_initializer(minval=-0.08,maxval=0.08))
					b = tf.get_variable(name="b", shape=[self.layers, 4, self.state_size], initializer=tf.zeros_initializer())

					h_prev,c_prev = tf.unstack(prev)
					h_full, c_full = [], []

					inp = x

					for i in range(self.layers):

						with tf.name_scope("gates"):
							with tf.name_scope("ft"):
								ft = tf.sigmoid(tf.matmul(inp,W[i][0]) + tf.matmul(h_prev[i],U[i][0]) + b[i][0])
							with tf.name_scope("it"):
								it = tf.sigmoid(tf.matmul(inp,W[i][1]) + tf.matmul(h_prev[i],U[i][1]) + b[i][1])
							with tf.name_scope("ot"):
								ot = tf.sigmoid(tf.matmul(inp,W[i][2]) + tf.matmul(h_prev[i],U[i][2]) + b[i][2])
							with tf.name_scope("ct"):
								ct = tf.tanh(tf.matmul(inp,W[i][3]) + tf.matmul(h_prev[i],U[i][3]) + b[i][3])

						with tf.name_scope("c"):
							c = (ft * c_prev[i]) + (it * ct)
						with tf.name_scope("h"):
							h = ot * tf.tanh(c)

						h_full.append(h)
						c_full.append(c)

						inp = h

				return tf.stack([h_full,c_full])

			with tf.device(self.heavy_device):

				# This will end up being shape [batch_size,state_size] currently it is [batch_size, seq_length]
				x_ = tf.placeholder(shape=[None,None],dtype=tf.int64,name="x_")
				y_ = tf.placeholder(shape=[None],dtype=tf.int64,name="y_")
				initial_state = tf.placeholder(shape=[2,self.layers,None,self.state_size],dtype=tf.float32,name='initial_state')
				
				# Create embedding. This will be a trainable parameter.
				embedding = tf.get_variable("embedding",shape=[self.num_classes,self.state_size])
				# This will create a tensor size of [batch_size, seq_length, state_size] which we can feed into our graph
				inputs = tf.nn.embedding_lookup(embedding,x_)

				# We need to reshape inputs to have seq_length as its first dimension so tf.scan can run over it giving us [batch_size,state_size] at each time step
				inputs = tf.transpose(inputs,[1,0,2])

				# Now we can pass inputs into tf.scan
				# This will give us an output size of [seq_length, 2, num_layers, batch_size, state_size] 

				outputs = tf.scan(step,inputs,initial_state)


				# TODO: Expose this later
				last_state = outputs[-1]

				# We only want the hidden state on the last layer
				# This should be of size batch_size, seq_length, state_size
				# if we reshape the first two dimensions we can do a matrix multiply with our new weights to compute logits
				states = tf.transpose(outputs,[1,2,3,0,4])[0][-1]

				# Now we create our final hidden layer weights
				# Our Y value will in theory be size of [batch_size, seq_length * num_classes] except we will reshape this into a column vector 
				# and it will not be onehot because sparse softmax will handle this
				
				W_f = tf.get_variable(name="W_f",shape=[self.state_size,self.num_classes],initializer=tf.random_uniform_initializer(minval=-0.08,maxval=0.08))
				b_f = tf.get_variable(name="b_f",shape=[self.num_classes],initializer=tf.zeros_initializer())

				# reshape to size [batch_size * seq_length, state_size]
				logits = tf.matmul(tf.reshape(states,[-1,self.state_size]),W_f)

				# Create our predictions
				predictions = tf.nn.softmax(logits)

				# Because this is sparse softmax, y_ will become onehot and of shape [batch_size * seq_length, num_classes]
				# Remember it is already size [batch_length * seq_length] because each character is represented by an int index
				losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=y_)

				loss = tf.reduce_mean(losses)

				optimize = tf.train.AdagradOptimizer(0.1).minimize(loss)

			self.x_ = x_
			self.y_ = y_
			self.initial_state = initial_state
			self.predictions = predictions
			self.loss = loss
			self.optimize = optimize
			self.last_state = last_state

		print("Building Graph")
		__graph__()
		print("Done.")

	def train(self,train_step,iterations=2000,restore=False):
		# Called to train model
		with tf.device(self.light_device):
			saver = tf.train.Saver()

		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
			try:

				if restore:
	  				saver.restore(sess, tf.train.latest_checkpoint('./saves'))
				else:
					sess.run(tf.global_variables_initializer())

				i = 0
				while True:
					# Get our batch of random samples 
					# Now let's encode this in an embedding 
					x_sample, y_sample = train_step.__next__()

					batch_size = x_sample.shape[0]

					feed = {self.x_: x_sample,
							self.y_: y_sample.flatten(),
							self.initial_state: np.zeros(shape=(2,self.layers,batch_size,self.state_size),dtype=np.float32)}

					_,loss = sess.run([self.optimize,self.loss],feed_dict=feed)

					if (i%100 == 0):
						print("Loss: " + str(loss))
					if (i%1000 == 0):
						print("Saving model....")
						save_path = saver.save(sess, "./saves/model.ckpt")
					i += 1

			except KeyboardInterrupt:
				print("Interrupted... Saving model.")

			save_path = saver.save(sess, "./saves/model.ckpt")

	def generate(self,char2ix,ix2char,seq_length,restore=False):
		# Called to generate samples from trained model

		# create seed 

		seed = np.random.choice(list(char2ix.values()))

		out = ""

		with tf.device(self.light_device):

			saver = tf.train.Saver()

		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

			# init session
			if restore:
  				saver.restore(sess, tf.train.latest_checkpoint('./saves'))
			else:
				sess.run(tf.global_variables_initializer())

			initialize = False

			for i in range(seq_length):

				if initialize == False:
					# if it is the first letter in sequence, intialize with default hidden states
					feed = {
						self.x_: np.array([seed]).reshape(1,1),
						self.initial_state: np.zeros(shape=(2,self.layers,1,self.state_size),dtype=np.float32)
					}
				else:				
					# otherwise, we need to intialize previous hidden state with the returned last state	
					feed = {
						self.x_: np.array([seed]).reshape(1,1),
						self.initial_state: last_state
					}


				predictions,last_state = sess.run([self.predictions,self.last_state],feed_dict = feed)

				initialize = True

				seed = np.random.choice(range(len(ix2char)),p=np.ravel(predictions))

				out += ix2char[seed]

			return out









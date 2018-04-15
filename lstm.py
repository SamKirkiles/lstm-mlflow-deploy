import tensorflow as tf
import numpy as np
import os 
import pickle

class LSTM:

	vocab_size = None
	data_size = None
	num_layers = 3
	hidden_size = 512

	char_to_ix = {}
	ix_to_char = {}

	# Train mode hiddne state
	h_state_prev = None
	c_state_prev = None

	# Test mode hidden state
	h_predict_prev = None
	c_predict_prev = None
 
	def __init__(self):
		pass

	def test(self,data,seq_length=1000,gpu=False,restore=False):
		chars = list(set(data))
		self.data_size,self.vocab_size = len(data),len(chars)

		if restore:
			with open('./saves/ix_to_char.pickle', 'rb') as f:
				self.ix_to_char = pickle.load(f)
			with open('./saves/char_to_ix.pickle', 'rb') as f:
				self.char_to_ix = pickle.load(f)
		else:
			self.char_to_ix = {ch:i for i,ch in enumerate(chars)}
			self.ix_to_char = {i:ch for i,ch in enumerate(chars)}


		Wout = tf.get_variable(name="Wout",shape=[self.vocab_size,self.hidden_size],dtype=tf.float32,initializer=tf.random_uniform_initializer(minval=-0.08,maxval=0.08))

		# Define graph
		if gpu == True:
			device = "/gpu:1"
		else:
			device = "/cpu:0"


		with tf.name_scope("predict_hidden"):
			with tf.device(device):	
				h_predict_placeholder = tf.placeholder(shape=[self.num_layers, self.hidden_size, 1],dtype=tf.float32,name="h_predict")
				c_predict_placeholder = tf.placeholder(shape=[self.num_layers, self.hidden_size, 1],dtype=tf.float32,name="c_predict")
				x_predict_placeholder = tf.placeholder(shape=[self.vocab_size,1],dtype=tf.float32,name="x_predict")

				hidden_states_pred = tf.stack([h_predict_placeholder,c_predict_placeholder])

				state = self.lstm_cell(hidden_states_pred,x_predict_placeholder,train=False)

				pred_hidden_state = tf.unstack(state)
				h,c = tf.unstack(state)
				h = h[-1]
				c = c[-1]

				h_pred_out = tf.matmul(Wout, h)	
				h_softmax = tf.reshape(tf.nn.softmax(tf.squeeze(h_pred_out)),[self.vocab_size,1])

		with tf.device("/cpu:0"):
			saver = tf.train.Saver()

		with tf.Session() as sess:
			if restore:
				print(tf.train.latest_checkpoint('./saves'))
				saver.restore(sess, tf.train.latest_checkpoint('./saves'))
			else:
				sess.run(tf.global_variables_initializer())

			self.h_predict_prev = np.zeros(shape=(self.num_layers,self.hidden_size,1),dtype=np.float32)
			self.c_predict_prev = np.zeros(shape=(self.num_layers,self.hidden_size,1),dtype=np.float32)

			one_hot_init = np.zeros((self.vocab_size,1),dtype=np.float32)
			one_hot_init[self.char_to_ix['a']] = 1
			
			out = ""

			for i in range(1000):

				feed_pred = {h_predict_placeholder: self.h_predict_prev,
							 c_predict_placeholder: self.c_predict_prev,
							 x_predict_placeholder: one_hot_init}

				softmax_pred,(h_pred,c_pred) = sess.run([h_softmax,pred_hidden_state],feed_dict=feed_pred)

				self.h_predict_prev = h_pred
				self.c_predict_prev = c_pred

				one_hot_n = np.random.choice(range(self.vocab_size),p=np.ravel(softmax_pred))
				one_hot_init = np.zeros((self.vocab_size),dtype=np.float32)
				one_hot_init[one_hot_n] = 1
				one_hot_init = np.reshape(one_hot_init,(self.vocab_size,1))

				out += self.ix_to_char[one_hot_n]
			return out



	def train(self,data,gpu=False,restore=False):

		run_id = np.random.randint(1000)
		seq_length = 50

		print("Training with run id: " + str(run_id))

		chars = list(set(data))
		self.data_size,self.vocab_size = len(data),len(chars)

		if restore:
			with open('./saves/ix_to_char.pickle') as f:
			    self.ix_to_col = json.load(f)
			with open('./saves/char_to_ix.pickle') as f:
			    self.col_to_ix = json.load(f)
		else:
			with open('./saves/ix_to_char.pickle', 'wb') as f:
				self.ix_to_char = {i:ch for i,ch in enumerate(chars)}
				pickle.dump(self.ix_to_char, f)
			with open('./saves/char_to_ix.pickle', 'wb') as f:
				self.char_to_ix = {ch:i for i,ch in enumerate(chars)}
				pickle.dump(self.char_to_ix, f)

		# Define graph (This is one lstm cell but we want multiple cells)	

		if gpu == True:
			device = "/gpu:0"
		else:
			device = "/cpu:0"

		with tf.device(device):

			with tf.name_scope("hidden_states"):

				h_prev_placeholder = tf.placeholder(shape=[self.num_layers,self.hidden_size,1],dtype=tf.float32,name="h_prev")
				c_prev_placeholder = tf.placeholder(shape=[self.num_layers,self.hidden_size,1],dtype=tf.float32,name="c_prev")
				hidden_states = tf.stack([h_prev_placeholder,c_prev_placeholder])

			inputs_placeholder = tf.placeholder(shape=[seq_length,self.vocab_size],dtype=tf.float32,name="batch")
			labels_placeholder = tf.placeholder(shape=[seq_length,self.vocab_size],dtype=tf.float32,name="batch")

			# this will be [25,2,vocab_size]
			batch = tf.scan(self.lstm_cell,inputs_placeholder,initializer=hidden_states)
			h_outputs,c_outputs = tf.unstack(batch,axis=1)

			h_outputs = h_outputs[:,-1,:,:]
			c_outputs = c_outputs[:,-1,:,:]

			Wout = tf.get_variable(name="Wout",shape=[self.vocab_size,self.hidden_size],dtype=tf.float32,initializer=tf.random_uniform_initializer(minval=-0.08,maxval=0.08))

			h_outputs = tf.squeeze(h_outputs,axis=2)

			h_new = tf.transpose(tf.matmul(Wout, tf.transpose(h_outputs)))

			softmax = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_placeholder,logits=h_new)
			
			loss = tf.reduce_mean(softmax)
			optimize = tf.train.AdagradOptimizer(0.1).minimize(loss)

			hidden_state = tf.unstack(batch,axis=1)

		# Test Graph
		if gpu == True:
			device = "/gpu:1"
		else:
			device = "/cpu:0"

		with tf.name_scope("predict_hidden"):
			with tf.device(device):	

				h_predict_placeholder = tf.placeholder(shape=[self.num_layers, self.hidden_size, 1],dtype=tf.float32,name="h_predict")
				c_predict_placeholder = tf.placeholder(shape=[self.num_layers, self.hidden_size, 1],dtype=tf.float32,name="c_predict")
				hidden_states_pred = tf.stack([h_predict_placeholder,c_predict_placeholder])

				x_predict_placeholder = tf.placeholder(shape=[self.vocab_size,1],dtype=tf.float32,name="x_predict")

				state = self.lstm_cell(hidden_states_pred,x_predict_placeholder,train=False)

				state_unstack = tf.unstack(state)
				h,c = tf.unstack(state)
				h = h[-1]
				c = c[-1]

				h_pred_out = tf.matmul(Wout, h)	
				h_softmax = tf.reshape(tf.nn.softmax(tf.squeeze(h_pred_out)),[self.vocab_size,1])

		with tf.device("/cpu:0"):
			saver = tf.train.Saver()


		with tf.Session() as sess:

			# Create summary writer
			train_writer = tf.summary.FileWriter('out_graph/train_' + str(run_id), sess.graph)

			i = 0
			j = 0			

			# initialize all veraibles
			if restore:
  				saver.restore(sess, tf.train.latest_checkpoint('./saves'))
			else:
				sess.run(tf.global_variables_initializer())

			self.h_state_prev = np.zeros(shape=(self.num_layers,self.hidden_size,1),dtype=np.float32)
			self.c_state_prev = np.zeros(shape=(self.num_layers,self.hidden_size,1),dtype=np.float32)
			loss_output = 0

			while True:

				if i + seq_length + 1 >= len(data) or j == 0:
					self.h_state_prev = np.zeros(shape=(self.num_layers,self.hidden_size,1),dtype=np.float32)
					self.c_state_prev = np.zeros(shape=(self.num_layers,self.hidden_size,1),dtype=np.float32)
					i = 0


				if j%1000 == 0:
					self.h_predict_prev = np.zeros(shape=(self.num_layers,self.hidden_size,1),dtype=np.float32)
					self.c_predict_prev = np.zeros(shape=(self.num_layers,self.hidden_size,1),dtype=np.float32)

					one_hot_init = np.zeros((self.vocab_size,1),dtype=np.float32)
					one_hot_init[self.char_to_ix['a']] = 1
					
					out = ""

					for i in range(300):

						feed_pred = {h_predict_placeholder: self.h_predict_prev,
									 c_predict_placeholder: self.c_predict_prev,
									 x_predict_placeholder: one_hot_init}

						softmax_pred,(h_pred,c_pred) = sess.run([h_softmax,state_unstack],feed_dict=feed_pred)

						self.h_predict_prev = h_pred
						self.c_predict_prev = c_pred


						one_hot_n = np.random.choice(range(self.vocab_size),p=np.ravel(softmax_pred))
						one_hot_init = np.zeros((self.vocab_size),dtype=np.float32)
						one_hot_init[one_hot_n] = 1
						one_hot_init = np.reshape(one_hot_init,(self.vocab_size,1))

						out += self.ix_to_char[one_hot_n]

					print("####### Loss: " + str(loss_output) + " ########")
					print(out)


					save_path = saver.save(sess, "./saves/model.ckpt",global_step=j)
					print("Model saved in path: %s" % save_path)


				inputs = np.array([self.char_to_ix[ch] for ch in data[i:i+seq_length]])
				targets = np.array([self.char_to_ix[ch] for ch in data[i+1:i+seq_length+1]])

				inputs_one_hot = np.zeros((inputs.shape[0],self.vocab_size),dtype=np.float32)
				inputs_one_hot[np.arange(inputs.shape[0]),inputs] = 1

				targets_one_hot = np.zeros((targets.shape[0],self.vocab_size),dtype=np.float32)
				targets_one_hot[np.arange(targets.shape[0]),targets] = 1


				feed = {inputs_placeholder:inputs_one_hot, 
						labels_placeholder:targets_one_hot, 
						h_prev_placeholder:self.h_state_prev, 
						c_prev_placeholder:self.c_state_prev}

				# Compute hidden states

				_,loss_output = sess.run([optimize, loss],feed_dict=feed)
				h_steps,c_steps= sess.run(hidden_state,feed_dict=feed)

				# set new hidden states
				self.h_state_prev = h_steps[-1]
				self.c_state_prev = c_steps[-1]

				i += seq_length
				j += 1
				print(j, end="\r", flush=True)


			train_writer.close()


	def lstm_cell(self,state,x,train=True):

		# This cell takes in previous hidden states of size (2,num_layers,vocab_size,1) and input of size (vocab_size)
		with tf.variable_scope("weights",reuse=tf.AUTO_REUSE):
			W_input = tf.get_variable(name="W_input",shape=[4, self.hidden_size, self.vocab_size],initializer=tf.random_uniform_initializer(minval=-0.08,maxval=0.08))
			U_input = tf.get_variable(name = "U_input",shape=[4, self.hidden_size, self.hidden_size],initializer=tf.random_uniform_initializer(minval=-0.08,maxval=0.08))
			b_input = tf.get_variable(name="b_input", shape=[4, self.hidden_size,1], initializer=tf.zeros_initializer())

			W = tf.get_variable(name="W",shape=[self.num_layers, 4, self.hidden_size, self.hidden_size],initializer=tf.random_uniform_initializer(minval=-0.08,maxval=0.08))
			U = tf.get_variable(name = "U",shape=[self.num_layers, 4, self.hidden_size, self.hidden_size],initializer=tf.random_uniform_initializer(minval=-0.08,maxval=0.08))
			b = tf.get_variable(name="b", shape=[self.num_layers, 4, self.hidden_size,1], initializer=tf.zeros_initializer())

		with tf.name_scope("LSTM_cell"):

			x = tf.reshape(x,[self.vocab_size,1])
			h_prev,c_prev = tf.unstack(state)
			inp = tf.layers.dropout(x,rate=0.5,training=train)

			h_full, c_full = [], []

			# First cell
			with tf.name_scope("input_gates"):
				with tf.name_scope("ft"):
					ft = tf.sigmoid(tf.matmul(W_input[0],inp) + tf.matmul(U_input[0],h_prev[0]) + b_input[0])
				with tf.name_scope("it"):
					it = tf.sigmoid(tf.matmul(W_input[1],inp) + tf.matmul(U_input[1],h_prev[0]) + b_input[1])
				with tf.name_scope("ot"):
					ot = tf.sigmoid(tf.matmul(W_input[2],inp) + tf.matmul(U_input[2],h_prev[0]) + b_input[2])
				with tf.name_scope("ct"):
					ct = tf.tanh(tf.matmul(W_input[3],inp) + tf.matmul(U_input[3],h_prev[0]) + b_input[3])

			with tf.name_scope("c"):
				c = (ft * c_prev[0]) + (it * ct)
			with tf.name_scope("h"):
				h = ot * tf.tanh(c)

			h_full.append(h)
			c_full.append(c)


			for i in range(1,self.num_layers):

				inp = tf.layers.dropout(h,rate=0.5,training=train)

				with tf.name_scope("gates"):
					with tf.name_scope("ft"):
						ft = tf.sigmoid(tf.matmul(W[i][0],inp) + tf.matmul(U[i][0],h_prev[i]) + b[i][0])
					with tf.name_scope("it"):
						it = tf.sigmoid(tf.matmul(W[i][1],inp) + tf.matmul(U[i][1],h_prev[i]) + b[i][1])
					with tf.name_scope("ot"):
						ot = tf.sigmoid(tf.matmul(W[i][2],inp) + tf.matmul(U[i][2],h_prev[i]) + b[i][2])
					with tf.name_scope("ct"):
						ct = tf.tanh(tf.matmul(W[i][3],inp) + tf.matmul(U[i][3],h_prev[i]) + b[i][3])

				with tf.name_scope("c"):
					c = (ft * c_prev[i]) + (it * ct)
				with tf.name_scope("h"):
					h = ot * tf.tanh(c)

				h_full.append(h)
				c_full.append(c)

			return tf.stack([h_full,c_full])


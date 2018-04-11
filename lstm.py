import tensorflow as tf
import numpy as np

class LSTM:

	vocab_size = None
	data_size = None
	hidden_size = 1024 * 4

	char_to_ix = {}
	ix_to_char = {}

	h_state_prev = None
	c_state_prev = None

	h_predict_prev = None
	c_predict_prev = None

	def __init__(self):
		pass


	def train(self,data):

		run_id = np.random.randint(1000)
		seq_length = 25

		print("Training with run id: " + str(run_id))

		# Create data tables
		chars = list(set(data))
		self.data_size,self.vocab_size = len(data),len(chars)

		self.char_to_ix = {ch:i for i,ch in enumerate(chars)}
		self.ix_to_char = {i:ch for i,ch in enumerate(chars)}

		# Define graph (This is one lstm cell but we want multiple cells)
		with tf.device("/gpu:0"):

			with tf.name_scope("hidden_states"):

				h_prev_placeholder = tf.placeholder(shape=[self.hidden_size,1],dtype=tf.float32,name="h_prev")
				c_prev_placeholder = tf.placeholder(shape=[self.hidden_size,1],dtype=tf.float32,name="c_prev")
				hidden_states = tf.stack([h_prev_placeholder,c_prev_placeholder])

			inputs_placeholder = tf.placeholder(shape=[seq_length,self.vocab_size],dtype=tf.float32,name="batch")
			labels_placeholder = tf.placeholder(shape=[seq_length,self.vocab_size],dtype=tf.float32,name="batch")

			# this will be [25,2,vocab_size]
			batch = tf.scan(self.lstm_cell,inputs_placeholder,initializer=hidden_states)
			h_outputs,c_outputs = tf.unstack(batch,axis=1)


			Wout = tf.get_variable(name="Wout",shape=[self.vocab_size,self.hidden_size],dtype=tf.float32)

			h_outputs = tf.squeeze(h_outputs,axis=2)

			h_new = tf.transpose(tf.matmul(Wout, tf.transpose(h_outputs)))

			softmax = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_placeholder,logits=h_new)
			
			loss = tf.reduce_mean(softmax)
			optimize = tf.train.AdagradOptimizer(0.1).minimize(loss)

			hidden_state = tf.unstack(batch,axis=1)
			# now compute loss and what not


		with tf.name_scope("predict_hidden"):
			with tf.device("/cpu:0"):	

				h_predict_placeholder = tf.placeholder(shape=[self.hidden_size,1],dtype=tf.float32,name="h_predict")
				c_predict_placeholder = tf.placeholder(shape=[self.hidden_size,1],dtype=tf.float32,name="c_predict")
				hidden_states_pred = tf.stack([h_predict_placeholder,c_predict_placeholder])

				x_predict_placeholder = tf.placeholder(shape=[self.vocab_size,1],dtype=tf.float32,name="x_predict")

				state = self.lstm_cell(hidden_states_pred,x_predict_placeholder)
				state_unstack = tf.unstack(state)
				h,c = tf.unstack(state)
				h_pred_out = tf.matmul(Wout, h)	
				h_softmax = tf.reshape(tf.nn.softmax(tf.squeeze(h_pred_out)),[self.vocab_size,1])

		with tf.Session() as sess:

			# Create summary writer
			train_writer = tf.summary.FileWriter('out_graph/train_' + str(run_id), sess.graph)

			i = 0
			j = 0			

			# initialize all veraibles
			sess.run(tf.global_variables_initializer())

			self.h_state_prev = np.zeros(shape=(self.hidden_size,1),dtype=np.float32)
			self.c_state_prev = np.zeros(shape=(self.hidden_size,1),dtype=np.float32)
			loss_output = 0

			while True:

				if i + seq_length + 1 >= len(data) or j == 0:
					self.h_state_prev = np.zeros(shape=(self.hidden_size,1),dtype=np.float32)
					self.c_state_prev = np.zeros(shape=(self.hidden_size,1),dtype=np.float32)
					i = 0


				if j%1000 == 0:
					self.h_predict_prev = np.zeros(shape=(self.hidden_size,1),dtype=np.float32)
					self.c_predict_prev = np.zeros(shape=(self.hidden_size,1),dtype=np.float32)

					one_hot_init = np.zeros((self.vocab_size,1),dtype=np.float32)
					one_hot_init[self.char_to_ix['a']] = 1
					
					out = ""

					for i in range(200):

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

			train_writer.close()


	def lstm_cell(self,state,x):


		with tf.variable_scope("weights",reuse=tf.AUTO_REUSE):

			Wfx = tf.get_variable(name="Wfx",shape=[self.hidden_size,self.vocab_size])
			Wfh = tf.get_variable(name = "Wfh",shape=[self.hidden_size,self.hidden_size])
			bf = tf.get_variable(name="bf", shape=[self.hidden_size,1], initializer=tf.zeros_initializer())

			Wix = tf.get_variable(name="Wix",shape=[self.hidden_size,self.vocab_size])
			Wih = tf.get_variable(name = "Wih",shape=[self.hidden_size,self.hidden_size])
			bi = tf.get_variable(name="bi", shape=[self.hidden_size,1], initializer=tf.zeros_initializer())

			Wcx = tf.get_variable(name="Wcx",shape=[self.hidden_size,self.vocab_size])
			Wch = tf.get_variable(name = "Wch",shape=[self.hidden_size,self.hidden_size])
			bc = tf.get_variable(name="bc", shape=[self.vocab_size,1], initializer=tf.zeros_initializer())

			Wox = tf.get_variable(name="Wox",shape=[self.hidden_size,self.vocab_size])
			Woh = tf.get_variable(name = "Woh",shape=[self.hidden_size,self.hidden_size])
			bo = tf.get_variable(name="bo", shape=[self.hidden_size,1], initializer=tf.zeros_initializer())

		with tf.name_scope("LSTM_cell"):

			print(x)
			x = tf.reshape(x,[self.vocab_size,1])
			h_prev,c_prev = tf.unstack(state)

			with tf.name_scope("gates"):
				with tf.name_scope("ft"):
					ft = tf.sigmoid(tf.matmul(Wfx,x) + tf.matmul(Wfh,h_prev) + bf)
				with tf.name_scope("it"):
					it = tf.sigmoid(tf.matmul(Wix,x) + tf.matmul(Wih,h_prev) + bi)
				with tf.name_scope("ot"):
					ot = tf.sigmoid(tf.matmul(Wox,x) + tf.matmul(Woh,h_prev) + bi)
				with tf.name_scope("ct"):
					ct = tf.tanh(tf.matmul(Wcx,x) + tf.matmul(Wch,h_prev) + bo)
		
			with tf.name_scope("c"):
				c = (ft * c_prev) + (it * ct)
			with tf.name_scope("h"):
				h = ot * tf.tanh(c)

			return tf.stack([h,c])





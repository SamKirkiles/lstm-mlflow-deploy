
import tensorflow as tf
import numpy as np

class LSTM:

	weights = {}

	vocab_size = None

	h_prev = None
	c_prev = None

	char_to_ix = {}
	ix_to_char = {}

	def __init__(self, vocab_size, lookup):
		print("Creating LSTM")

		self.vocab_size = vocab_size

		self.char_to_ix = lookup["char_to_ix"]
		self.ix_to_char = lookup["ix_to_char"]

		self.h_prev = tf.zeros([self.vocab_size],dtype=tf.float32)
		self.c_prev = tf.zeros([self.vocab_size],dtype=tf.float32)

		self.init_weights()


	def train(self,data):

		with tf.Session() as sess:

			print("Done")


			seq_length = np.random.randint(25)

			# Create placeholders
			with tf.name_scope("batch_inputs"):
				one_hot_p = tf.placeholder(dtype=tf.float32,shape=[None,self.vocab_size],name="inputs")

			with tf.name_scope("batch_targets"):
				one_hot_targets_p = tf.placeholder(dtype=tf.float32,shape=[None,self.vocab_size],name="targets")


			h = self.process_batch(one_hot_p,one_hot_targets_p)

			optimize = tf.train.AdagradOptimizer(0.1).minimize(h)

			train_writer = tf.summary.FileWriter('out_graph/train_' + str(np.random.randint(1000)), sess.graph)

			sess.run(tf.global_variables_initializer())

			

			i = 0
			j = 0

			while True:

				if i + seq_length + 1 >= len(data) or j == 0:
					seq_prev = np.zeros((self.vocab_size,1))
					i = 0

				inputs = np.array([self.char_to_ix[ch] for ch in data[i:i+seq_length]])
				targets = np.array([self.char_to_ix[ch] for ch in data[i+1:i+seq_length+1]])


				one_hot = np.zeros((inputs.shape[0],self.vocab_size),dtype=np.float32)
				one_hot[np.arange(inputs.shape[0]),inputs] = 1
				
				one_hot_targets = np.zeros((targets.shape[0],self.vocab_size),dtype=np.float32)
				one_hot_targets[np.arange(targets.shape[0]),targets] = 1

				predictions,loss = sess.run([optimize,h],feed_dict={one_hot_p:one_hot,one_hot_targets_p:one_hot_targets})
				

				print(loss)

				if j%100 == 0:
					self.predict(sess)

				i += seq_length
				j+=1

			train_writer.close()

	def init_weights(self):

		# Initialize all the weights for the lstm
		with tf.name_scope("weights"):

			weight_names = ["Wfx","Wfh","Wix","Wih","Wcx","Wch","Wox","Woh"]
			bias_names = ["bf","bi","bc","bo"]

			for key in weight_names:
				self.weights[key] = tf.Variable(initial_value=tf.random_normal([self.vocab_size,self.vocab_size],dtype=tf.float32),name=key)* 0.01

			for key in bias_names:
				self.weights[key] = tf.Variable(initial_value=tf.zeros([self.vocab_size,1],dtype=tf.float32),name=key)


	def process_batch(self,one_hot,one_hot_targets):

		
		hs_scan = tf.zeros([tf.shape(one_hot)[0],self.vocab_size],dtype=tf.float32)
		C_scan = tf.zeros([tf.shape(one_hot)[0],self.vocab_size],dtype=tf.float32)

		(x,hs,c) = tf.scan(self.lstm_cell,(one_hot,hs_scan,C_scan),initializer=(one_hot[0],self.h_prev,self.c_prev))
		output = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_targets,logits=hs)
	
		loss = tf.reduce_mean(output)

		self.h_prev = hs[-1]
		self.c_prev = c[-1]

		return loss

	def predict(self,sess):

		one_hot = np.zeros((self.vocab_size),dtype=np.float32)
		one_hot[self.char_to_ix['a']] = 1
		one_hot = tf.convert_to_tensor(one_hot)
		
		h = tf.zeros([self.vocab_size],dtype=np.float32)
		c = tf.zeros([self.vocab_size],dtype=np.float32)

		out = ""

		for i in range(25):
			(x,h,c) = self.lstm_cell((one_hot,h,c),(one_hot,h,c))
			h_softmax = tf.nn.softmax(h)

			h_out = sess.run(h_softmax)
			one_hot_n = np.random.choice(range(self.vocab_size),p=h_out)
			one_hot = np.zeros((self.vocab_size),dtype=np.float32)
			one_hot[one_hot_n] = 1
			one_hot = tf.convert_to_tensor(one_hot)

			out += self.ix_to_char[one_hot_n]

		print(out)

	def lstm_cell(self,prev,current):

			# Runs one character through the lstm cell
			# takes in a tuple (x-1,h-1,C-1) and (x,h,c) to predict h
			# To be used with tf.scan

			(x_prev,h_prev,C_prev) = prev
			(x,h_current,C_current) = current
				
			x = tf.reshape(x,[self.vocab_size,1])
			h_prev = tf.reshape(h_prev,[self.vocab_size,1])
			C_prev = tf.reshape(C_prev,[self.vocab_size,1])

			Wfx = self.weights["Wfx"]
			Wfh = self.weights["Wfh"]
			bf = self.weights["bf"]

			Wix = self.weights["Wix"]
			Wih = self.weights["Wih"]
			bi = self.weights["bi"]		

			Wcx = self.weights["Wcx"]
			Wch = self.weights["Wch"]
			bc = self.weights["bc"]		

			Wox = self.weights["Wox"]
			Woh = self.weights["Woh"]
			bo = self.weights["bo"]		

			with tf.name_scope("lstm_cell"):

				fg = tf.sigmoid(tf.matmul(Wfx,x) + tf.matmul(Wfh,h_prev) + bf)
				ig =  tf.sigmoid(tf.matmul(Wix,x) + tf.matmul(Wih,h_prev) + bi)
				og = tf.sigmoid(tf.matmul(Wox, x) + tf.matmul(Woh,h_prev)  + bo)

				C_tilde = tf.tanh(tf.matmul(Wcx,x) + tf.matmul(Wch,h_prev) + bc)
				C = tf.multiply(fg, C_prev) + tf.multiply(ig, C_tilde)

				h = tf.multiply(og, tf.tanh(C))

			return (tf.squeeze(x),tf.squeeze(h),tf.squeeze(C))



import tensorflow as tf
import numpy as np


with tf.Session() as sess:

	filenames = ['input.txt']
	dataset = tf.data.TextLineDataset(filenames)
	
	dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(25))

	iterator = dataset.make_initializable_iterator()
	next_element = iterator.get_next()

	for _ in range(1):
		sess.run(iterator.initializer)
		while True:
			try:
				print("Start")
				print(sess.run(next_element)[0])
				print("end")
			except tf.errors.OutOfRangeError:
				break
